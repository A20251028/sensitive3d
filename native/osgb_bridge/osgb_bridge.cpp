// osgb_bridge: OpenSceneGraph helper used by the sensitive3d Python package.
//
// Commands
//   export <in.osgb> <out_dir>
//       Dump every osg::Geometry (triangulated) and every texture image of one
//       OSGB file into <out_dir>/manifest.json plus raw little-endian arrays.
//   patch  <in.osgb> <patch_dir> <out.osgb>
//       Re-read <in.osgb>, replace the geometries / images listed in
//       <patch_dir>/patch.txt and write <out.osgb>.  Everything else in the
//       scene graph (PagedLOD hierarchy, ranges, file names, state sets) is
//       kept.  Images that are not replaced are re-embedded with their
//       original encoded bytes (no JPEG generation loss).
//   info   <file_list.txt>
//       Print one JSON line per listed file: per-geometry world bounds,
//       triangle count and LOD flags plus referenced child files.  Images
//       are not decoded, so this is a cheap way to scan a whole dataset.
//   build  <build.txt> <out.osgb>
//       Build an OSGB file from a small text description (used to create
//       synthetic test datasets with a PagedLOD hierarchy).
//
// Array files written/read by this tool
//   *.f32  float32, *.u32  uint32, little endian, no header.
//   textures are raw 8-bit RGB / RGBA, rows top-down (row 0 is v == 1).

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/LOD>
#include <osg/MatrixTransform>
#include <osg/PagedLOD>
#include <osg/Texture2D>
#include <osg/TriangleIndexFunctor>
#include <osgDB/FileNameUtils>
#include <osgDB/FileUtils>
#include <osgDB/ReadFile>
#include <osgDB/ReaderWriter>
#include <osgDB/Registry>
#include <osgDB/WriteFile>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Image byte capture: a proxy ReaderWriter registered in front of the real
// image plugins remembers the encoded bytes of every inline image so that
// untouched textures can be written back bit-exactly.
// ---------------------------------------------------------------------------

struct CapturedImage {
    std::string ext;
    std::string bytes;
};

static std::map<const osg::Image*, CapturedImage> g_captured;
// When set, inline images are not decoded (used by the fast "info" scan).
static bool g_skipImageDecode = false;

class CapturingImageRW : public osgDB::ReaderWriter {
public:
    explicit CapturingImageRW(const std::string& ext) : _ext(ext) {
        supportsExtension(ext, "sensitive3d capturing image proxy");
    }
    const char* className() const override { return "sensitive3d capturing image proxy"; }

    osgDB::ReaderWriter* real() const {
        if (_real.valid()) return _real.get();
        osgDB::Registry* reg = osgDB::Registry::instance();
        reg->loadLibrary(reg->createLibraryNameForExtension(_ext));
        osgDB::Registry::ReaderWriterList& list = reg->getReaderWriterList();
        for (auto& rw : list) {
            if (dynamic_cast<CapturingImageRW*>(rw.get())) continue;
            if (rw->acceptsExtension(_ext)) {
                _real = rw.get();
                break;
            }
        }
        return _real.get();
    }

    ReadResult readImage(std::istream& fin, const Options* opt) const override {
        std::string bytes((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
        if (g_skipImageDecode) {
            osg::ref_ptr<osg::Image> stub = new osg::Image;
            stub->allocateImage(1, 1, 1, GL_RGB, GL_UNSIGNED_BYTE);
            return stub.release();
        }
        osgDB::ReaderWriter* rw = real();
        if (!rw) return ReadResult::FILE_NOT_HANDLED;
        std::istringstream ss(bytes);
        ReadResult rr = rw->readImage(ss, opt);
        if (rr.validImage()) g_captured[rr.getImage()] = CapturedImage{_ext, bytes};
        return rr;
    }

    ReadResult readImage(const std::string& file, const Options* opt) const override {
        osgDB::ReaderWriter* rw = real();
        if (!rw) return ReadResult::FILE_NOT_HANDLED;
        std::string path = osgDB::findDataFile(file, opt);
        if (path.empty()) return ReadResult::FILE_NOT_FOUND;
        std::ifstream in(path, std::ios::binary);
        if (!in) return ReadResult::FILE_NOT_FOUND;
        return readImage(in, opt);
    }

    WriteResult writeImage(const osg::Image& img, const std::string& file, const Options* opt) const override {
        osgDB::ReaderWriter* rw = real();
        if (!rw) return WriteResult::FILE_NOT_HANDLED;
        return rw->writeImage(img, file, opt);
    }

    WriteResult writeImage(const osg::Image& img, std::ostream& out, const Options* opt) const override {
        osgDB::ReaderWriter* rw = real();
        if (!rw) return WriteResult::FILE_NOT_HANDLED;
        return rw->writeImage(img, out, opt);
    }

private:
    std::string _ext;
    mutable osg::observer_ptr<osgDB::ReaderWriter> _real;
};

static void registerCapturingReaders() {
    const char* exts[] = {"jpg", "jpeg", "png", "dds", "tga", "bmp", "tif", "tiff", "gif", "ktx"};
    for (const char* e : exts) osgDB::Registry::instance()->addReaderWriter(new CapturingImageRW(e));
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

static std::string jsonEscape(const std::string& s) {
    std::ostringstream o;
    for (unsigned char c : s) {
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if (c < 0x20) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c << std::dec;
                else o << c;
        }
    }
    return o.str();
}

template <typename T>
static bool writeRaw(const fs::path& p, const std::vector<T>& v) {
    std::ofstream out(p, std::ios::binary);
    if (!out) return false;
    if (!v.empty()) out.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
    return bool(out);
}

template <typename T>
static bool readRaw(const fs::path& p, std::vector<T>& v) {
    std::ifstream in(p, std::ios::binary | std::ios::ate);
    if (!in) return false;
    std::streamsize n = in.tellg();
    in.seekg(0);
    v.resize(size_t(n) / sizeof(T));
    if (n > 0) in.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(T));
    return bool(in);
}

static std::string readFileBytes(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

static osg::Texture2D* findTexture2D(osg::StateSet* ss) {
    if (!ss) return nullptr;
    return dynamic_cast<osg::Texture2D*>(ss->getTextureAttribute(0, osg::StateAttribute::TEXTURE));
}

// "Detail" of an LOD child: a larger value means a finer level of detail.
static double childDetail(const osg::LOD& lod, unsigned i) {
    if (i >= lod.getNumRanges()) return 0.0;
    if (lod.getRangeMode() == osg::LOD::PIXEL_SIZE_ON_SCREEN) return lod.getMinRange(i);
    return -lod.getMaxRange(i);
}

// ---------------------------------------------------------------------------
// Scene collection (shared by export and patch so that indices match)
// ---------------------------------------------------------------------------

struct GeomRecord {
    osg::Geometry* geom = nullptr;
    osg::Matrixd matrix;
    osg::Texture2D* texture = nullptr;
    int imageIndex = -1;
    bool hasFiner = false;
    int lodIndex = -1;
    int lodChild = -1;
    int depth = 0;
};

struct LodRecord {
    osg::LOD* lod = nullptr;
    bool paged = false;
};

class Collector : public osg::NodeVisitor {
public:
    Collector() : osg::NodeVisitor(TRAVERSE_ALL_CHILDREN) {}

    std::vector<GeomRecord> geoms;
    std::vector<osg::Image*> images;
    std::vector<LodRecord> lods;

    void apply(osg::LOD& lod) override {
        int idx = int(lods.size());
        LodRecord rec;
        rec.lod = &lod;
        rec.paged = dynamic_cast<osg::PagedLOD*>(&lod) != nullptr;
        lods.push_back(rec);
        osg::PagedLOD* plod = dynamic_cast<osg::PagedLOD*>(&lod);
        for (unsigned i = 0; i < lod.getNumChildren(); ++i) {
            bool finer = false;
            double di = childDetail(lod, i);
            unsigned n = std::max<unsigned>(lod.getNumRanges(), lod.getNumChildren());
            for (unsigned j = 0; j < n; ++j) {
                if (j == i) continue;
                bool exists = j < lod.getNumChildren();
                if (!exists && plod && j < plod->getNumFileNames() && !plod->getFileName(j).empty()) exists = true;
                if (exists && childDetail(lod, j) > di) finer = true;
            }
            _finer.push_back(finer);
            _lod.push_back({idx, int(i)});
            lod.getChild(i)->accept(*this);
            _lod.pop_back();
            _finer.pop_back();
        }
    }

    void apply(osg::Geometry& g) override {
        GeomRecord r;
        r.geom = &g;
        r.matrix = osg::computeLocalToWorld(getNodePath());
        const osg::NodePath& path = getNodePath();
        for (auto it = path.rbegin(); it != path.rend() && !r.texture; ++it) r.texture = findTexture2D((*it)->getStateSet());
        if (r.texture && r.texture->getImage()) {
            osg::Image* img = r.texture->getImage();
            auto found = _imageIndex.find(img);
            if (found == _imageIndex.end()) {
                r.imageIndex = int(images.size());
                _imageIndex[img] = r.imageIndex;
                images.push_back(img);
            } else {
                r.imageIndex = found->second;
            }
        }
        for (bool f : _finer) r.hasFiner = r.hasFiner || f;
        if (!_lod.empty()) {
            r.lodIndex = _lod.back().first;
            r.lodChild = _lod.back().second;
        }
        r.depth = int(_lod.size());
        geoms.push_back(r);
    }

private:
    std::vector<bool> _finer;
    std::vector<std::pair<int, int>> _lod;
    std::map<osg::Image*, int> _imageIndex;
};

// ---------------------------------------------------------------------------
// Geometry extraction
// ---------------------------------------------------------------------------

struct TriCollector {
    std::vector<uint32_t>* out = nullptr;
    void operator()(unsigned a, unsigned b, unsigned c) {
        if (a == b || b == c || a == c) return;
        out->push_back(a);
        out->push_back(b);
        out->push_back(c);
    }
};

static bool getVertices(osg::Geometry& g, std::vector<float>& v) {
    osg::Array* a = g.getVertexArray();
    if (auto* f = dynamic_cast<osg::Vec3Array*>(a)) {
        v.resize(f->size() * 3);
        for (size_t i = 0; i < f->size(); ++i) {
            v[i * 3] = (*f)[i].x();
            v[i * 3 + 1] = (*f)[i].y();
            v[i * 3 + 2] = (*f)[i].z();
        }
        return true;
    }
    if (auto* d = dynamic_cast<osg::Vec3dArray*>(a)) {
        v.resize(d->size() * 3);
        for (size_t i = 0; i < d->size(); ++i) {
            v[i * 3] = float((*d)[i].x());
            v[i * 3 + 1] = float((*d)[i].y());
            v[i * 3 + 2] = float((*d)[i].z());
        }
        return true;
    }
    return false;
}

static bool getTexcoords(osg::Geometry& g, size_t nverts, std::vector<float>& uv) {
    auto* t = dynamic_cast<osg::Vec2Array*>(g.getTexCoordArray(0));
    if (!t || t->size() != nverts) return false;
    uv.resize(t->size() * 2);
    for (size_t i = 0; i < t->size(); ++i) {
        uv[i * 2] = (*t)[i].x();
        uv[i * 2 + 1] = (*t)[i].y();
    }
    return true;
}

static bool getNormals(osg::Geometry& g, size_t nverts, std::vector<float>& n) {
    auto* a = dynamic_cast<osg::Vec3Array*>(g.getNormalArray());
    if (!a || a->getBinding() != osg::Array::BIND_PER_VERTEX || a->size() != nverts) return false;
    n.resize(a->size() * 3);
    for (size_t i = 0; i < a->size(); ++i) {
        n[i * 3] = (*a)[i].x();
        n[i * 3 + 1] = (*a)[i].y();
        n[i * 3 + 2] = (*a)[i].z();
    }
    return true;
}

static bool getColors(osg::Geometry& g, size_t nverts, std::vector<float>& c) {
    osg::Array* a = g.getColorArray();
    if (!a || a->getBinding() != osg::Array::BIND_PER_VERTEX || a->getNumElements() != nverts) return false;
    c.resize(nverts * 4);
    if (auto* f = dynamic_cast<osg::Vec4Array*>(a)) {
        for (size_t i = 0; i < nverts; ++i)
            for (int k = 0; k < 4; ++k) c[i * 4 + k] = (*f)[i][k];
        return true;
    }
    if (auto* u = dynamic_cast<osg::Vec4ubArray*>(a)) {
        for (size_t i = 0; i < nverts; ++i)
            for (int k = 0; k < 4; ++k) c[i * 4 + k] = (*u)[i][k] / 255.0f;
        return true;
    }
    return false;
}

static bool dumpImage(const osg::Image* img, const fs::path& p, int& w, int& h, int& channels) {
    w = img->s();
    h = img->t();
    GLenum pf = img->getPixelFormat();
    bool alpha = pf == GL_RGBA || pf == GL_BGRA || pf == GL_LUMINANCE_ALPHA || pf == GL_COMPRESSED_RGBA_S3TC_DXT1_EXT ||
                 pf == GL_COMPRESSED_RGBA_S3TC_DXT3_EXT || pf == GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
    channels = alpha ? 4 : 3;
    std::vector<uint8_t> out(size_t(w) * h * channels);
    bool simple = !img->isCompressed() && img->getDataType() == GL_UNSIGNED_BYTE && img->r() <= 1 &&
                  (pf == GL_RGB || pf == GL_RGBA || pf == GL_BGR || pf == GL_BGRA || pf == GL_LUMINANCE ||
                   pf == GL_LUMINANCE_ALPHA);
    for (int y = 0; y < h; ++y) {
        int srcRow = h - 1 - y;  // texture row 0 is v == 0, output row 0 is v == 1
        uint8_t* dst = out.data() + size_t(y) * w * channels;
        if (simple) {
            const uint8_t* src = img->data(0, srcRow);
            int sc = osg::Image::computeNumComponents(pf);
            for (int x = 0; x < w; ++x) {
                const uint8_t* s = src + size_t(x) * sc;
                uint8_t r, g, b, a = 255;
                if (pf == GL_LUMINANCE || pf == GL_LUMINANCE_ALPHA) {
                    r = g = b = s[0];
                    if (pf == GL_LUMINANCE_ALPHA) a = s[1];
                } else if (pf == GL_BGR || pf == GL_BGRA) {
                    b = s[0]; g = s[1]; r = s[2];
                    if (pf == GL_BGRA) a = s[3];
                } else {
                    r = s[0]; g = s[1]; b = s[2];
                    if (pf == GL_RGBA) a = s[3];
                }
                dst[x * channels] = r;
                dst[x * channels + 1] = g;
                dst[x * channels + 2] = b;
                if (channels == 4) dst[x * channels + 3] = a;
            }
        } else {
            for (int x = 0; x < w; ++x) {
                osg::Vec4 c = img->getColor(x, srcRow);
                for (int k = 0; k < channels; ++k) {
                    float v = std::min(1.0f, std::max(0.0f, c[k]));
                    dst[x * channels + k] = uint8_t(v * 255.0f + 0.5f);
                }
            }
        }
    }
    return writeRaw(p, out);
}

// ---------------------------------------------------------------------------
// export
// ---------------------------------------------------------------------------

static int cmdExport(const std::string& in, const fs::path& outDir) {
    osg::ref_ptr<osg::Node> root = osgDB::readRefNodeFile(in);
    if (!root) {
        std::cerr << "cannot read " << in << std::endl;
        return 2;
    }
    fs::create_directories(outDir);
    Collector col;
    root->accept(col);

    std::ofstream js(outDir / "manifest.json");
    js << std::setprecision(17);
    js << "{\n  \"source\": \"" << jsonEscape(in) << "\",\n  \"geometries\": [\n";
    for (size_t gi = 0; gi < col.geoms.size(); ++gi) {
        GeomRecord& r = col.geoms[gi];
        std::vector<float> v, uv, n, c;
        std::vector<uint32_t> tris;
        bool hasV = getVertices(*r.geom, v);
        size_t nv = v.size() / 3;
        bool hasUV = hasV && getTexcoords(*r.geom, nv, uv);
        bool hasN = hasV && getNormals(*r.geom, nv, n);
        bool hasC = hasV && getColors(*r.geom, nv, c);
        if (hasV) {
            osg::TriangleIndexFunctor<TriCollector> tif;
            tif.out = &tris;
            r.geom->accept(tif);
        }
        std::string base = "g" + std::to_string(gi);
        if (hasV) writeRaw(outDir / (base + "_v.f32"), v);
        if (hasUV) writeRaw(outDir / (base + "_uv.f32"), uv);
        if (hasN) writeRaw(outDir / (base + "_n.f32"), n);
        if (hasC) writeRaw(outDir / (base + "_c.f32"), c);
        if (hasV) writeRaw(outDir / (base + "_t.u32"), tris);
        js << "    {\"index\": " << gi << ", \"name\": \"" << jsonEscape(r.geom->getName()) << "\""
           << ", \"num_vertices\": " << nv << ", \"num_triangles\": " << tris.size() / 3
           << ", \"vertices\": " << (hasV ? "\"" + base + "_v.f32\"" : "null")
           << ", \"texcoords\": " << (hasUV ? "\"" + base + "_uv.f32\"" : "null")
           << ", \"normals\": " << (hasN ? "\"" + base + "_n.f32\"" : "null")
           << ", \"colors\": " << (hasC ? "\"" + base + "_c.f32\"" : "null")
           << ", \"triangles\": " << (hasV ? "\"" + base + "_t.u32\"" : "null")
           << ", \"texture\": " << r.imageIndex << ", \"has_finer\": " << (r.hasFiner ? "true" : "false")
           << ", \"lod\": " << r.lodIndex << ", \"lod_child\": " << r.lodChild << ", \"depth\": " << r.depth
           << ", \"matrix\": [";
        const double* m = r.matrix.ptr();
        for (int k = 0; k < 16; ++k) js << (k ? ", " : "") << m[k];
        js << "]}" << (gi + 1 < col.geoms.size() ? "," : "") << "\n";
    }
    js << "  ],\n  \"textures\": [\n";
    for (size_t ii = 0; ii < col.images.size(); ++ii) {
        osg::Image* img = col.images[ii];
        int w = 0, h = 0, ch = 0;
        std::string fname = "tex" + std::to_string(ii) + ".raw";
        bool ok = dumpImage(img, outDir / fname, w, h, ch);
        auto cap = g_captured.find(img);
        std::string encoding = cap != g_captured.end() ? cap->second.ext : (img->isCompressed() ? "compressed" : "raw");
        js << "    {\"index\": " << ii << ", \"file\": " << (ok ? "\"" + fname + "\"" : "null") << ", \"width\": " << w
           << ", \"height\": " << h << ", \"channels\": " << ch << ", \"name\": \"" << jsonEscape(img->getFileName())
           << "\", \"encoding\": \"" << encoding << "\"}" << (ii + 1 < col.images.size() ? "," : "") << "\n";
    }
    js << "  ],\n  \"lods\": [\n";
    for (size_t li = 0; li < col.lods.size(); ++li) {
        osg::LOD* lod = col.lods[li].lod;
        osg::PagedLOD* plod = dynamic_cast<osg::PagedLOD*>(lod);
        const osg::Vec3d& c = lod->getCenter();
        js << "    {\"index\": " << li << ", \"paged\": " << (plod ? "true" : "false") << ", \"center\": [" << c.x()
           << ", " << c.y() << ", " << c.z() << "], \"radius\": " << lod->getRadius() << ", \"range_mode\": \""
           << (lod->getRangeMode() == osg::LOD::PIXEL_SIZE_ON_SCREEN ? "pixel" : "distance") << "\", \"children\": [";
        unsigned n = std::max<unsigned>(lod->getNumRanges(), lod->getNumChildren());
        for (unsigned i = 0; i < n; ++i) {
            std::string file = plod && i < plod->getNumFileNames() ? plod->getFileName(i) : "";
            double mn = i < lod->getNumRanges() ? lod->getMinRange(i) : 0.0;
            double mx = i < lod->getNumRanges() ? lod->getMaxRange(i) : 0.0;
            js << (i ? ", " : "") << "{\"file\": \"" << jsonEscape(file) << "\", \"loaded\": "
               << (i < lod->getNumChildren() ? "true" : "false") << ", \"min\": " << mn << ", \"max\": " << mx << "}";
        }
        js << "]}" << (li + 1 < col.lods.size() ? "," : "") << "\n";
    }
    js << "  ]\n}\n";
    return js ? 0 : 3;
}

// ---------------------------------------------------------------------------
// Writing with embedded image files
// ---------------------------------------------------------------------------

class ImageGatherer : public osg::NodeVisitor {
public:
    ImageGatherer() : osg::NodeVisitor(TRAVERSE_ALL_CHILDREN) {}
    std::vector<osg::Image*> images;
    std::set<osg::Image*> seen;
    std::vector<osg::LOD*> lods;
    void apply(osg::Node& node) override {
        gather(node.getStateSet());
        if (auto* lod = dynamic_cast<osg::LOD*>(&node)) lods.push_back(lod);
        traverse(node);
    }

private:
    void gather(osg::StateSet* ss) {
        if (!ss) return;
        for (unsigned u = 0; u < ss->getTextureAttributeList().size(); ++u) {
            auto* tex = dynamic_cast<osg::Texture*>(ss->getTextureAttribute(u, osg::StateAttribute::TEXTURE));
            if (!tex) continue;
            for (unsigned k = 0; k < tex->getNumImages(); ++k) {
                osg::Image* img = tex->getImage(k);
                if (img && seen.insert(img).second) images.push_back(img);
            }
        }
    }
};

// replacement: encoded bytes + extension to use for a given image
struct ImageBytes {
    std::string ext;
    std::string bytes;
};

static bool writeNodeEmbedded(osg::Node* root, const std::string& outPath,
                              const std::map<const osg::Image*, ImageBytes>& overrides) {
    ImageGatherer gat;
    root->accept(gat);

    // PagedLOD database paths are re-derived from the file location on load;
    // never bake absolute input directories into the output.
    for (osg::LOD* lod : gat.lods)
        if (auto* plod = dynamic_cast<osg::PagedLOD*>(lod)) plod->setDatabasePath("");

    fs::path outFile(outPath);
    if (outFile.has_parent_path()) fs::create_directories(outFile.parent_path());
    fs::path tmp = fs::path(outPath + ".s3dtmp");
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    std::set<std::string> used;
    int counter = 0;
    for (osg::Image* img : gat.images) {
        ImageBytes ib;
        auto ov = overrides.find(img);
        auto cap = g_captured.find(img);
        if (ov != overrides.end()) {
            ib = ov->second;
        } else if (cap != g_captured.end()) {
            ib.ext = cap->second.ext;
            ib.bytes = cap->second.bytes;
        } else {
            // raw inline image without original encoded bytes: encode losslessly
            ib.ext = img->isCompressed() ? "dds" : "png";
            fs::path enc = tmp / ("__encode." + ib.ext);
            if (!osgDB::writeImageFile(*img, enc.string())) {
                std::cerr << "cannot encode image " << img->getFileName() << std::endl;
                fs::remove_all(tmp);
                return false;
            }
            ib.bytes = readFileBytes(enc);
            fs::remove(enc);
        }
        std::string base = osgDB::getStrippedName(img->getFileName());
        if (base.empty()) base = "texture";
        std::string name = base + "." + ib.ext;
        while (used.count(name)) name = base + "_" + std::to_string(counter++) + "." + ib.ext;
        used.insert(name);
        std::ofstream o(tmp / name, std::ios::binary);
        o.write(ib.bytes.data(), ib.bytes.size());
        o.close();
        img->setFileName(name);
        img->setWriteHint(osg::Image::NO_PREFERENCE);
    }

    osg::ref_ptr<osgDB::Options> opts = new osgDB::Options("WriteImageHint=IncludeFile");
    opts->setDatabasePath(tmp.string());
    bool ok = osgDB::writeNodeFile(*root, outPath, opts.get());
    fs::remove_all(tmp);
    if (!ok) std::cerr << "cannot write " << outPath << std::endl;
    return ok;
}

// ---------------------------------------------------------------------------
// patch
// ---------------------------------------------------------------------------

static osg::ref_ptr<osg::Vec3Array> loadVec3(const fs::path& p) {
    std::vector<float> v;
    if (!readRaw(p, v)) return nullptr;
    osg::ref_ptr<osg::Vec3Array> a = new osg::Vec3Array(v.size() / 3);
    for (size_t i = 0; i < a->size(); ++i) (*a)[i].set(v[i * 3], v[i * 3 + 1], v[i * 3 + 2]);
    return a;
}

static osg::ref_ptr<osg::Vec2Array> loadVec2(const fs::path& p) {
    std::vector<float> v;
    if (!readRaw(p, v)) return nullptr;
    osg::ref_ptr<osg::Vec2Array> a = new osg::Vec2Array(v.size() / 2);
    for (size_t i = 0; i < a->size(); ++i) (*a)[i].set(v[i * 2], v[i * 2 + 1]);
    return a;
}

static osg::ref_ptr<osg::Vec4Array> loadVec4(const fs::path& p) {
    std::vector<float> v;
    if (!readRaw(p, v)) return nullptr;
    osg::ref_ptr<osg::Vec4Array> a = new osg::Vec4Array(v.size() / 4);
    for (size_t i = 0; i < a->size(); ++i) (*a)[i].set(v[i * 4], v[i * 4 + 1], v[i * 4 + 2], v[i * 4 + 3]);
    return a;
}

static void replaceGeometry(osg::Geometry& g, osg::Vec3Array* verts, osg::Vec2Array* uvs, osg::Vec3Array* normals,
                            osg::Vec4Array* colors, const std::vector<uint32_t>& tris) {
    size_t oldCount = g.getVertexArray() ? g.getVertexArray()->getNumElements() : 0;
    size_t nv = verts->size();
    g.setVertexArray(verts);
    if (uvs) g.setTexCoordArray(0, uvs, osg::Array::BIND_PER_VERTEX);
    else if (g.getTexCoordArray(0) && g.getTexCoordArray(0)->getNumElements() != nv) g.setTexCoordArray(0, nullptr);
    for (unsigned u = 1; u < g.getNumTexCoordArrays(); ++u)
        if (g.getTexCoordArray(u) && g.getTexCoordArray(u)->getNumElements() != nv) g.setTexCoordArray(u, nullptr);
    if (normals) g.setNormalArray(normals, osg::Array::BIND_PER_VERTEX);
    else if (g.getNormalArray() && g.getNormalArray()->getBinding() == osg::Array::BIND_PER_VERTEX &&
             (oldCount != nv || g.getNormalArray()->getNumElements() != nv))
        g.setNormalArray(nullptr);
    if (colors) g.setColorArray(colors, osg::Array::BIND_PER_VERTEX);
    else if (g.getColorArray() && g.getColorArray()->getBinding() == osg::Array::BIND_PER_VERTEX &&
             g.getColorArray()->getNumElements() != nv)
        g.setColorArray(nullptr);
    for (unsigned a = 0; a < g.getNumVertexAttribArrays(); ++a)
        if (g.getVertexAttribArray(a) && g.getVertexAttribArray(a)->getNumElements() != nv)
            g.setVertexAttribArray(a, nullptr);
    g.removePrimitiveSet(0, g.getNumPrimitiveSets());
    osg::ref_ptr<osg::DrawElementsUInt> de = new osg::DrawElementsUInt(GL_TRIANGLES, tris.begin(), tris.end());
    g.addPrimitiveSet(de.get());
    g.dirtyBound();
    g.dirtyGLObjects();
}

static int cmdPatch(const std::string& in, const fs::path& patchDir, const std::string& out) {
    osg::ref_ptr<osg::Node> root = osgDB::readRefNodeFile(in);
    if (!root) {
        std::cerr << "cannot read " << in << std::endl;
        return 2;
    }
    Collector col;
    root->accept(col);

    std::ifstream pf(patchDir / "patch.txt");
    if (!pf) {
        std::cerr << "missing patch.txt" << std::endl;
        return 2;
    }
    std::map<const osg::Image*, ImageBytes> overrides;
    std::string line;
    while (std::getline(pf, line)) {
        std::istringstream ls(line);
        std::string kind;
        if (!(ls >> kind) || kind[0] == '#') continue;
        if (kind == "geometry") {
            size_t idx;
            std::string vf, uvf, nf, cf, tf;
            ls >> idx >> vf >> uvf >> nf >> cf >> tf;
            if (idx >= col.geoms.size()) {
                std::cerr << "geometry index out of range: " << idx << std::endl;
                return 4;
            }
            osg::ref_ptr<osg::Vec3Array> v = loadVec3(patchDir / vf);
            osg::ref_ptr<osg::Vec2Array> uv = uvf == "-" ? nullptr : loadVec2(patchDir / uvf);
            osg::ref_ptr<osg::Vec3Array> n = nf == "-" ? nullptr : loadVec3(patchDir / nf);
            osg::ref_ptr<osg::Vec4Array> c = cf == "-" ? nullptr : loadVec4(patchDir / cf);
            std::vector<uint32_t> tris;
            if (!v || !readRaw(patchDir / tf, tris)) {
                std::cerr << "cannot read arrays for geometry " << idx << std::endl;
                return 4;
            }
            replaceGeometry(*col.geoms[idx].geom, v.get(), uv.get(), n.get(), c.get(), tris);
        } else if (kind == "texture") {
            size_t idx;
            std::string file;
            ls >> idx >> file;
            if (idx >= col.images.size()) {
                std::cerr << "texture index out of range: " << idx << std::endl;
                return 4;
            }
            ImageBytes ib;
            ib.ext = osgDB::convertToLowerCase(osgDB::getFileExtension(file));
            ib.bytes = readFileBytes(patchDir / file);
            if (ib.bytes.empty()) {
                std::cerr << "empty texture file " << file << std::endl;
                return 4;
            }
            osg::Image* img = col.images[idx];
            overrides[img] = ib;
            // keep the in-memory image consistent with the new data
            osg::ref_ptr<osg::Image> fresh = osgDB::readRefImageFile((patchDir / file).string());
            if (fresh) {
                std::string name = img->getFileName();
                img->allocateImage(fresh->s(), fresh->t(), fresh->r(), fresh->getPixelFormat(), fresh->getDataType(),
                                   fresh->getPacking());
                memcpy(img->data(), fresh->data(), fresh->getTotalSizeInBytes());
                img->setInternalTextureFormat(fresh->getInternalTextureFormat());
                img->setFileName(name);
            }
        }
    }
    return writeNodeEmbedded(root.get(), out, overrides) ? 0 : 5;
}

// ---------------------------------------------------------------------------
// build (synthetic datasets)
// ---------------------------------------------------------------------------

class Builder {
public:
    explicit Builder(const fs::path& base) : _base(base) {}

    osg::ref_ptr<osg::Node> parse(std::istream& in) {
        std::string tok;
        if (!(in >> tok)) return nullptr;
        return parseNode(tok, in);
    }

private:
    fs::path _base;
    std::map<std::string, osg::ref_ptr<osg::Texture2D>> _textures;

    osg::ref_ptr<osg::Node> parseNode(const std::string& tok, std::istream& in) {
        if (tok == "BEGIN_PAGEDLOD") {
            double cx, cy, cz, r;
            int mode;
            in >> cx >> cy >> cz >> r >> mode;
            osg::ref_ptr<osg::PagedLOD> p = new osg::PagedLOD;
            p->setCenterMode(osg::LOD::USER_DEFINED_CENTER);
            p->setCenter(osg::Vec3d(cx, cy, cz));
            p->setRadius(r);
            p->setRangeMode(mode ? osg::LOD::PIXEL_SIZE_ON_SCREEN : osg::LOD::DISTANCE_FROM_EYE_POINT);
            std::string t;
            unsigned idx = 0;
            while (in >> t && t != "END_PAGEDLOD") {
                if (t == "CHILD") {
                    double mn, mx;
                    std::string nt;
                    in >> mn >> mx >> nt;
                    osg::ref_ptr<osg::Node> child = parseNode(nt, in);
                    p->addChild(child.get(), float(mn), float(mx));
                    ++idx;
                } else if (t == "FILE_CHILD") {
                    std::string file;
                    double mn, mx;
                    in >> file >> mn >> mx;
                    p->setFileName(idx, file);
                    p->setRange(idx, float(mn), float(mx));
                    ++idx;
                }
            }
            return p;
        }
        if (tok == "BEGIN_GROUP" || tok == "BEGIN_MATRIX") {
            osg::ref_ptr<osg::Group> g;
            if (tok == "BEGIN_MATRIX") {
                double m[16];
                for (double& x : m) in >> x;
                osg::ref_ptr<osg::MatrixTransform> mt = new osg::MatrixTransform(osg::Matrixd(m));
                g = mt;
            } else {
                g = new osg::Group;
            }
            std::string end = tok == "BEGIN_GROUP" ? "END_GROUP" : "END_MATRIX";
            std::string t;
            while (in >> t && t != end) {
                osg::ref_ptr<osg::Node> c = parseNode(t, in);
                if (c) g->addChild(c.get());
            }
            return g;
        }
        if (tok == "BEGIN_GEODE") {
            osg::ref_ptr<osg::Geode> geode = new osg::Geode;
            std::string t;
            while (in >> t && t != "END_GEODE") {
                if (t != "GEOMETRY") continue;
                std::string vf, uvf, nf, tf, texf;
                in >> vf >> uvf >> nf >> tf >> texf;
                osg::ref_ptr<osg::Geometry> g = new osg::Geometry;
                osg::ref_ptr<osg::Vec3Array> v = loadVec3(_base / vf);
                std::vector<uint32_t> tris;
                readRaw(_base / tf, tris);
                osg::ref_ptr<osg::Vec2Array> uv = uvf == "-" ? nullptr : loadVec2(_base / uvf);
                osg::ref_ptr<osg::Vec3Array> n = nf == "-" ? nullptr : loadVec3(_base / nf);
                g->setUseVertexBufferObjects(true);
                replaceGeometry(*g, v.get(), uv.get(), n.get(), nullptr, tris);
                if (texf != "-") {
                    osg::StateSet* ss = g->getOrCreateStateSet();
                    ss->setTextureAttributeAndModes(0, texture(texf), osg::StateAttribute::ON);
                    ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
                }
                geode->addDrawable(g.get());
            }
            return geode;
        }
        std::cerr << "unexpected token " << tok << std::endl;
        return nullptr;
    }

    osg::Texture2D* texture(const std::string& file) {
        auto it = _textures.find(file);
        if (it != _textures.end()) return it->second.get();
        fs::path p = _base / file;
        osg::ref_ptr<osg::Image> img = osgDB::readRefImageFile(p.string());
        if (!img) {
            std::cerr << "cannot read texture " << p << std::endl;
            return nullptr;
        }
        img->setFileName(osgDB::getSimpleFileName(file));
        osg::ref_ptr<osg::Texture2D> tex = new osg::Texture2D(img.get());
        tex->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
        tex->setWrap(osg::Texture::WRAP_T, osg::Texture::CLAMP_TO_EDGE);
        tex->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR_MIPMAP_LINEAR);
        tex->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
        _textures[file] = tex;
        return tex.get();
    }
};

static int cmdBuild(const fs::path& desc, const std::string& out) {
    std::ifstream in(desc);
    if (!in) {
        std::cerr << "cannot read " << desc << std::endl;
        return 2;
    }
    Builder b(desc.parent_path());
    osg::ref_ptr<osg::Node> root = b.parse(in);
    if (!root) return 3;
    return writeNodeEmbedded(root.get(), out, {}) ? 0 : 5;
}

// ---------------------------------------------------------------------------
// info: fast structural scan of many files (images are not decoded)
// ---------------------------------------------------------------------------

static int cmdInfo(const fs::path& listFile) {
    g_skipImageDecode = true;
    std::ifstream in(listFile);
    if (!in) {
        std::cerr << "cannot read " << listFile << std::endl;
        return 2;
    }
    std::cout << std::setprecision(10);
    std::string path;
    while (std::getline(in, path)) {
        if (path.empty()) continue;
        osg::ref_ptr<osg::Node> root = osgDB::readRefNodeFile(path);
        std::cout << "{\"file\": \"" << jsonEscape(path) << "\", \"ok\": " << (root ? "true" : "false");
        if (root) {
            Collector col;
            root->accept(col);
            std::cout << ", \"geometries\": [";
            for (size_t gi = 0; gi < col.geoms.size(); ++gi) {
                GeomRecord& r = col.geoms[gi];
                std::vector<float> v;
                osg::BoundingBox bb;
                size_t ntri = 0;
                if (getVertices(*r.geom, v)) {
                    for (size_t i = 0; i < v.size(); i += 3)
                        bb.expandBy(osg::Vec3d(v[i], v[i + 1], v[i + 2]) * r.matrix);
                    std::vector<uint32_t> tris;
                    osg::TriangleIndexFunctor<TriCollector> tif;
                    tif.out = &tris;
                    r.geom->accept(tif);
                    ntri = tris.size() / 3;
                }
                std::cout << (gi ? ", " : "") << "{\"has_finer\": " << (r.hasFiner ? "true" : "false")
                          << ", \"num_triangles\": " << ntri << ", \"depth\": " << r.depth;
                if (bb.valid())
                    std::cout << ", \"min\": [" << bb.xMin() << ", " << bb.yMin() << ", " << bb.zMin() << "], \"max\": ["
                              << bb.xMax() << ", " << bb.yMax() << ", " << bb.zMax() << "]";
                std::cout << "}";
            }
            std::cout << "], \"children\": [";
            bool first = true;
            for (auto& l : col.lods) {
                auto* plod = dynamic_cast<osg::PagedLOD*>(l.lod);
                if (!plod) continue;
                for (unsigned i = 0; i < plod->getNumFileNames(); ++i) {
                    if (plod->getFileName(i).empty()) continue;
                    std::cout << (first ? "" : ", ") << "\"" << jsonEscape(plod->getFileName(i)) << "\"";
                    first = false;
                }
            }
            std::cout << "]";
        }
        std::cout << "}" << std::endl;
        g_captured.clear();
    }
    return 0;
}

int main(int argc, char** argv) {
    registerCapturingReaders();
    std::string cmd = argc > 1 ? argv[1] : "";
    try {
        if (cmd == "export" && argc == 4) return cmdExport(argv[2], argv[3]);
        if (cmd == "patch" && argc == 5) return cmdPatch(argv[2], argv[3], argv[4]);
        if (cmd == "build" && argc == 4) return cmdBuild(argv[2], argv[3]);
        if (cmd == "info" && argc == 3) return cmdInfo(argv[2]);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 10;
    }
    std::cerr << "usage:\n"
                 "  osgb_bridge export <in.osgb> <out_dir>\n"
                 "  osgb_bridge patch <in.osgb> <patch_dir> <out.osgb>\n"
                 "  osgb_bridge build <build.txt> <out.osgb>\n"
                 "  osgb_bridge info <file_list.txt>\n";
    return 1;
}
