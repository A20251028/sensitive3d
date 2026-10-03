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
//       Print exactly one JSON line per listed file (also for files that
//       cannot be read: "ok": false + "error"): per-geometry world bounds,
//       triangle count, LOD flags, UV / texture state, the file's images
//       (encoding and header-probed size, not decoded) and referenced child
//       files.  This is a cheap way to scan a whole dataset.
//   build  <build.txt> <out.osgb>
//       Build an OSGB file from a small text description (used to create
//       synthetic test datasets with a PagedLOD hierarchy).  A leading
//       EXTERNAL_IMAGES token stores textures as external file references.
//   version
//       Print OSG / bridge versions and where the plugins are looked up.
//   selftest <tmp_dir>
//       Write / read JPEG, PNG and a small PagedLOD .osgb and check that the
//       exported pixels and vertices match.  Exit code 0 only when all pass.
//
// Array files written/read by this tool
//   *.f32  float32, *.u32  uint32, little endian, no header.
//   textures are raw 8-bit RGB / RGBA, rows top-down (row 0 is v == 1).

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/LOD>
#include <osg/MatrixTransform>
#include <osg/Notify>
#include <osg/PagedLOD>
#include <osg/Texture2D>
#include <osg/TriangleIndexFunctor>
#include <osg/Version>
#include <osgDB/FileNameUtils>
#include <osgDB/FileUtils>
#include <osgDB/fstream>
#include <osgDB/ReadFile>
#include <osgDB/ReaderWriter>
#include <osgDB/Registry>
#include <osgDB/WriteFile>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    bool external = false;  // read from a file next to the .osgb, not embedded
};

static std::map<const osg::Image*, CapturedImage> g_captured;
// External image names that could not be found (for error messages).
static std::set<std::string> g_missingImageFiles;
// Images that failed to load are replaced by an empty placeholder (no
// pixels) instead of null, so that image indices are the same in info,
// export and patch and every failed texture can be reported by name.
static std::map<const osg::Image*, std::string> g_failedImages;
// When set, inline images are not decoded (used by the fast "info" scan).
static bool g_skipImageDecode = false;

static std::string pluginHint(const std::string& ext);

static osgDB::ReaderWriter::ReadResult placeholderImage(const std::string& error) {
    osg::ref_ptr<osg::Image> img = new osg::Image;
    g_failedImages[img.get()] = error;
    return img.release();
}

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
        return decode(bytes, opt, false);
    }

    ReadResult readImage(const std::string& file, const Options* opt) const override {
        // The registry offers every file to every reader: only handle our own
        // extension so that a .png is never pushed through the JPEG proxy.
        if (!acceptsExtension(osgDB::getLowerCaseFileExtension(file))) return ReadResult::FILE_NOT_HANDLED;
        std::string path = osgDB::findDataFile(file, opt);
        std::ifstream in;
        if (!path.empty()) in.open(path, std::ios::binary);
        if (path.empty() || !in) {
            g_missingImageFiles.insert(file);
            return placeholderImage("external image file not found");
        }
        std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return decode(bytes, opt, true);
    }

    WriteResult writeImage(const osg::Image& img, const std::string& file, const Options* opt) const override {
        if (!acceptsExtension(osgDB::getLowerCaseFileExtension(file))) return WriteResult::FILE_NOT_HANDLED;
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

    ReadResult decode(const std::string& bytes, const Options* opt, bool external) const {
        if (g_skipImageDecode) {
            // keep the encoded bytes so that "info" can probe the header
            osg::ref_ptr<osg::Image> stub = new osg::Image;
            stub->allocateImage(1, 1, 1, GL_RGB, GL_UNSIGNED_BYTE);
            g_captured[stub.get()] = CapturedImage{_ext, bytes, external};
            return stub.release();
        }
        std::string where = external ? "external " : "embedded ";
        osgDB::ReaderWriter* rw = real();
        if (!rw) return placeholderImage("cannot decode " + where + _ext + " image: " + pluginHint(_ext));
        std::istringstream ss(bytes);
        ReadResult rr = rw->readImage(ss, opt);
        if (!rr.validImage() || !rr.getImage()->data())
            return placeholderImage("cannot decode " + where + _ext + " data (" + std::to_string(bytes.size()) +
                                    " bytes, " + rr.statusMessage() + ")");
        g_captured[rr.getImage()] = CapturedImage{_ext, bytes, external};
        return rr;
    }
};

static void registerCapturingReaders() {
    const char* exts[] = {"jpg", "jpeg", "png", "dds", "tga", "bmp", "tif", "tiff", "gif", "ktx"};
    for (const char* e : exts) osgDB::Registry::instance()->addReaderWriter(new CapturingImageRW(e));
}

// OSG prints NOTICE / INFO messages to stdout by default, which would corrupt
// the JSON written by "info", "version" and "selftest".
class StderrNotifyHandler : public osg::NotifyHandler {
public:
    void notify(osg::NotifySeverity, const char* message) override { fputs(message, stderr); }
};

// ---------------------------------------------------------------------------
// Encoded image header probing (no decoding)
// ---------------------------------------------------------------------------

struct ImageProbe {
    std::string format;  // detected from the magic bytes ("" when unknown)
    int width = -1;
    int height = -1;
    bool ok = false;
    std::string error;
};

static std::string normalizeImageExt(std::string ext) {
    ext = osgDB::convertToLowerCase(ext);
    if (ext == "jpeg" || ext == "jpe") return "jpg";
    if (ext == "tif") return "tiff";
    return ext;
}

static unsigned be16(const unsigned char* p) { return (unsigned(p[0]) << 8) | p[1]; }
static uint32_t be32(const unsigned char* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
}
static uint32_t le32(const unsigned char* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

// Width / height of a JPEG from its first SOFn segment.
static bool probeJpeg(const unsigned char* b, size_t n, int& w, int& h) {
    size_t i = 2;
    while (i + 4 <= n) {
        if (b[i] != 0xFF) return false;
        unsigned char m = b[i + 1];
        if (m == 0xFF) {  // fill byte
            ++i;
            continue;
        }
        if (m == 0xD8 || m == 0x01 || (m >= 0xD0 && m <= 0xD7)) {  // markers without length
            i += 2;
            continue;
        }
        if (m == 0xD9 || m == 0xDA) return false;  // EOI / SOS before any SOF
        unsigned len = be16(b + i + 2);
        if (len < 2) return false;
        if (m >= 0xC0 && m <= 0xCF && m != 0xC4 && m != 0xC8 && m != 0xCC) {
            if (i + 9 > n) return false;
            h = int(be16(b + i + 5));
            w = int(be16(b + i + 7));
            return w > 0 && h > 0;
        }
        i += 2 + len;
    }
    return false;
}

static ImageProbe probeImageBytes(const std::string& bytes, const std::string& extIn) {
    ImageProbe p;
    std::string ext = normalizeImageExt(extIn);
    const unsigned char* b = reinterpret_cast<const unsigned char*>(bytes.data());
    size_t n = bytes.size();
    static const unsigned char pngSig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (n >= 3 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF) p.format = "jpg";
    else if (n >= 8 && memcmp(b, pngSig, 8) == 0) p.format = "png";
    else if (n >= 4 && memcmp(b, "DDS ", 4) == 0) p.format = "dds";
    else if (n >= 4 && memcmp(b, "GIF8", 4) == 0) p.format = "gif";
    else if (n >= 2 && b[0] == 'B' && b[1] == 'M') p.format = "bmp";
    else if (n >= 4 && (memcmp(b, "II*\0", 4) == 0 || memcmp(b, "MM\0*", 4) == 0)) p.format = "tiff";
    else if (n >= 12 && memcmp(b, "\xABKTX 11\xBB\r\n\x1A\n", 12) == 0) p.format = "ktx";

    if (n == 0) {
        p.error = "empty image data";
        return p;
    }
    if (p.format == "jpg") {
        if (!probeJpeg(b, n, p.width, p.height)) {
            p.width = p.height = -1;
            p.error = "JPEG header has no frame (SOF) marker before the scan data (truncated or corrupt)";
            return p;
        }
    } else if (p.format == "png") {
        if (n < 24 || memcmp(b + 12, "IHDR", 4) != 0) {
            p.error = "PNG without IHDR header (truncated or corrupt)";
            return p;
        }
        p.width = int(std::min<uint32_t>(be32(b + 16), 0x7fffffff));
        p.height = int(std::min<uint32_t>(be32(b + 20), 0x7fffffff));
    } else if (p.format == "dds") {
        if (n < 20) {
            p.error = "DDS header truncated";
            return p;
        }
        p.height = int(std::min<uint32_t>(le32(b + 12), 0x7fffffff));
        p.width = int(std::min<uint32_t>(le32(b + 16), 0x7fffffff));
    }
    // formats we can verify must carry their own magic bytes
    bool checkable = ext == "jpg" || ext == "png" || ext == "dds";
    if (checkable && p.format != ext) {
        p.error = "image data is not a valid " + ext + (p.format.empty() ? "" : " (looks like " + p.format + ")");
        return p;
    }
    if ((p.format == "jpg" || p.format == "png" || p.format == "dds") && (p.width <= 0 || p.height <= 0)) {
        p.error = "invalid image size in header";
        return p;
    }
    p.ok = true;
    return p;
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

// An image that failed to load (missing external file, undecodable inline
// data) is still created by the OSGB reader, but without pixels.
static bool imageHasData(const osg::Image* img) {
    return img && img->data() && img->s() > 0 && img->t() > 0 && img->r() > 0;
}

static std::string describeMissingImage(const osg::Image* img) {
    std::string name = img ? img->getFileName() : "";
    std::string shown = name.empty() ? std::string("<unnamed image>") : name;
    auto failed = g_failedImages.find(img);
    if (failed != g_failedImages.end()) return shown + ": " + failed->second;
    if (!name.empty() && g_missingImageFiles.count(name)) return shown + ": external image file not found";
    return shown + ": image has no pixel data";
}

static std::string pluginHint(const std::string& ext) {
    osgDB::Registry* reg = osgDB::Registry::instance();
    std::string lib = reg->createLibraryNameForExtension(ext);
    std::string found = osgDB::findLibraryFile(lib);
    if (!found.empty()) return "plugin " + found;
    std::string dir = osgDB::getFilePath(lib);
    return "plugin " + lib + " not found in the OSG library path; set OSG_LIBRARY_PATH to the directory that contains " +
           (dir.empty() ? lib : dir) + " (see 'osgb_bridge version')";
}

// Read a node file and explain failures (missing file, no plugin, parse error).
static osg::ref_ptr<osg::Node> readNodeChecked(const std::string& path, std::string& error) {
    std::error_code ec;
    if (!fs::is_regular_file(fs::path(path), ec)) {
        error = (fs::exists(fs::path(path), ec) ? "not a regular file: " : "file not found: ") + path;
        return nullptr;
    }
    osgDB::Registry* reg = osgDB::Registry::instance();
    osgDB::ReaderWriter::ReadResult rr = reg->readNode(path, reg->getOptions());
    if (rr.validNode()) return osg::ref_ptr<osg::Node>(rr.getNode());
    error = "cannot read " + path + ": " + rr.statusMessage();
    std::string ext = osgDB::getLowerCaseFileExtension(path);
    if (!reg->getReaderWriterForExtension(ext)) error += " (no reader for ." + ext + ": " + pluginHint(ext) + ")";
    return nullptr;
}

static std::string jsonStringList(const std::vector<std::string>& v) {
    std::string o = "[";
    for (size_t i = 0; i < v.size(); ++i) o += (i ? ", \"" : "\"") + jsonEscape(v[i]) + "\"";
    return o + "]";
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

// A geometry is bound to a Texture2D whose image is null or has no pixels.
static bool textureMissing(const GeomRecord& r, const Collector& col) {
    if (!r.texture) return false;
    if (r.imageIndex < 0) return true;
    return !imageHasData(col.images[size_t(r.imageIndex)]);
}

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

// Convert an image to 8-bit RGB / RGBA, rows top-down (row 0 is v == 1).
static void imageToRGB(const osg::Image* img, std::vector<uint8_t>& out, int& w, int& h, int& channels) {
    w = img->s();
    h = img->t();
    GLenum pf = img->getPixelFormat();
    bool alpha = pf == GL_RGBA || pf == GL_BGRA || pf == GL_LUMINANCE_ALPHA || pf == GL_COMPRESSED_RGBA_S3TC_DXT1_EXT ||
                 pf == GL_COMPRESSED_RGBA_S3TC_DXT3_EXT || pf == GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
    channels = alpha ? 4 : 3;
    out.assign(size_t(w) * h * channels, 0);
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
}

static bool dumpImage(const osg::Image* img, const fs::path& p, int& w, int& h, int& channels) {
    std::vector<uint8_t> out;
    imageToRGB(img, out, w, h, channels);
    return writeRaw(p, out);
}

// ---------------------------------------------------------------------------
// export
// ---------------------------------------------------------------------------

static int cmdExport(const std::string& in, const fs::path& outDir) {
    std::string readError;
    osg::ref_ptr<osg::Node> root = readNodeChecked(in, readError);
    if (!root) {
        std::cerr << readError << std::endl;
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
           << ", \"texture\": " << r.imageIndex << ", \"texture_missing\": " << (textureMissing(r, col) ? "true" : "false")
           << ", \"has_finer\": " << (r.hasFiner ? "true" : "false")
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
        auto cap = g_captured.find(img);
        std::string encoding = cap != g_captured.end() ? cap->second.ext : (img->isCompressed() ? "compressed" : "raw");
        // never write a broken / empty raw file for an image that did not decode
        std::string error;
        bool ok = false;
        if (!imageHasData(img)) {
            error = describeMissingImage(img);
            encoding = "missing";
        } else if (!(ok = dumpImage(img, outDir / fname, w, h, ch))) {
            error = "cannot write " + fname;
            std::error_code ec;
            fs::remove(outDir / fname, ec);
        }
        if (!ok) {
            w = h = ch = 0;
            std::cerr << "texture " << ii << " (" << img->getFileName() << "): " << error << std::endl;
        }
        js << "    {\"index\": " << ii << ", \"file\": " << (ok ? "\"" + fname + "\"" : "null") << ", \"width\": " << w
           << ", \"height\": " << h << ", \"channels\": " << ch << ", \"name\": \"" << jsonEscape(img->getFileName())
           << "\", \"encoding\": \"" << encoding << "\", \"valid\": " << (ok ? "true" : "false");
        if (!ok) js << ", \"error\": \"" << jsonEscape(error) << "\"";
        js << "}" << (ii + 1 < col.images.size() ? "," : "") << "\n";
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

// externalImages: store image file names only (WriteImageHint=UseExternal)
// instead of embedding the encoded bytes; used by "build" for test files.
static bool writeNodeEmbedded(osg::Node* root, const std::string& outPath,
                              const std::map<const osg::Image*, ImageBytes>& overrides, bool externalImages = false) {
    ImageGatherer gat;
    root->accept(gat);
    if (externalImages) gat.images.clear();

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
        } else if (!imageHasData(img)) {
            std::cerr << "refusing to write: texture " << describeMissingImage(img) << std::endl;
            fs::remove_all(tmp);
            return false;
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
        if (getenv("S3D_DEBUG"))
            std::cerr << "embed " << img << " as " << name << " (" << ib.bytes.size() << " bytes, "
                      << (ov != overrides.end() ? "override" : cap != g_captured.end() ? "captured" : "encoded") << ")" << std::endl;
    }

    osg::ref_ptr<osgDB::Options> opts =
        new osgDB::Options(externalImages ? "WriteImageHint=UseExternal" : "WriteImageHint=IncludeFile");
    opts->setDatabasePath(tmp.string());
    // Write through the stream interface: writing by file name would put the
    // output directory in front of the search path, so an unrelated image
    // with the same name next to the output file would be embedded instead.
    bool ok = false;
    osgDB::ReaderWriter* rw = osgDB::Registry::instance()->getReaderWriterForExtension("osgb");
    if (rw) {
        osgDB::ofstream fout(outPath.c_str(), std::ios::out | std::ios::binary);
        if (fout) {
            ok = rw->writeNode(*root, fout, opts.get()).success();
            fout.close();
            ok = ok && bool(fout);
        }
    }
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
    std::string readError;
    osg::ref_ptr<osg::Node> root = readNodeChecked(in, readError);
    if (!root) {
        std::cerr << readError << std::endl;
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
            if (getenv("S3D_DEBUG")) std::cerr << "override " << img << " <- " << file << " (" << ib.bytes.size() << " bytes)" << std::endl;
            // keep the in-memory image consistent with the new data
            osg::ref_ptr<osg::Image> fresh = osgDB::readRefImageFile((patchDir / file).string());
            if (imageHasData(fresh.get())) {
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

    // Leading directives: EXTERNAL_IMAGES (reference textures by file name
    // instead of embedding them).
    bool externalImages = false;

    osg::ref_ptr<osg::Node> parse(std::istream& in) {
        std::string tok;
        while (in >> tok) {
            if (tok == "EXTERNAL_IMAGES") {
                externalImages = true;
                continue;
            }
            return parseNode(tok, in);
        }
        return nullptr;
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
        if (!imageHasData(img.get())) {
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
    return writeNodeEmbedded(root.get(), out, {}, b.externalImages) ? 0 : 5;
}

// ---------------------------------------------------------------------------
// info: fast structural scan of many files (images are not decoded)
// ---------------------------------------------------------------------------

struct InfoImage {
    std::string name;
    std::string encoding;  // jpg / png / ... | raw | compressed | external | missing
    std::string format;    // detected encoded format, or the file extension
    bool external = false;
    int width = -1;
    int height = -1;
    uint64_t bytes = 0;
    bool ok = false;
    std::string error;
};

struct InfoGeom {
    bool hasFiner = false;
    size_t numTriangles = 0;
    int depth = 0;
    osg::BoundingBoxd bb;
    bool hasUV = false;
    int image = -1;
    bool textureMissing = false;
};

struct InfoResult {
    std::string file;
    bool ok = false;
    std::string error;
    std::vector<InfoGeom> geoms;
    std::vector<InfoImage> images;
    std::vector<std::string> children;
};

static InfoImage describeInfoImage(const osg::Image* img) {
    InfoImage r;
    r.name = img->getFileName();
    auto cap = g_captured.find(img);
    if (cap != g_captured.end()) {
        const CapturedImage& c = cap->second;
        ImageProbe p = probeImageBytes(c.bytes, c.ext);
        r.external = c.external;
        r.format = p.format.empty() ? normalizeImageExt(c.ext) : p.format;
        r.encoding = c.external ? "external" : normalizeImageExt(c.ext);
        r.width = p.width;
        r.height = p.height;
        r.bytes = c.bytes.size();
        r.ok = p.ok;
        if (!p.ok) r.error = (r.name.empty() ? std::string("<unnamed image>") : r.name) + ": " + p.error;
    } else if (imageHasData(img)) {
        r.encoding = r.format = img->isCompressed() ? "compressed" : "raw";
        r.width = img->s();
        r.height = img->t();
        r.bytes = img->getTotalSizeInBytes();
        r.ok = true;
    } else {
        r.encoding = "missing";
        r.format = normalizeImageExt(osgDB::getFileExtension(r.name));
        r.external = g_missingImageFiles.count(r.name) > 0;
        r.error = describeMissingImage(img);
    }
    return r;
}

static InfoResult scanFile(const std::string& path) {
    InfoResult res;
    res.file = path;
    if (path.empty()) {
        res.error = "empty path";
        return res;
    }
    osg::ref_ptr<osg::Node> root = readNodeChecked(path, res.error);
    if (!root) return res;
    res.ok = true;
    Collector col;
    root->accept(col);
    for (osg::Image* img : col.images) res.images.push_back(describeInfoImage(img));
    for (GeomRecord& r : col.geoms) {
        InfoGeom g;
        g.hasFiner = r.hasFiner;
        g.depth = r.depth;
        g.image = r.imageIndex;
        g.textureMissing = r.texture && (r.imageIndex < 0 || !res.images[size_t(r.imageIndex)].ok);
        std::vector<float> v;
        if (getVertices(*r.geom, v)) {
            for (size_t i = 0; i < v.size(); i += 3) g.bb.expandBy(osg::Vec3d(v[i], v[i + 1], v[i + 2]) * r.matrix);
            std::vector<uint32_t> tris;
            osg::TriangleIndexFunctor<TriCollector> tif;
            tif.out = &tris;
            r.geom->accept(tif);
            g.numTriangles = tris.size() / 3;
            auto* t = dynamic_cast<osg::Vec2Array*>(r.geom->getTexCoordArray(0));
            g.hasUV = t && t->size() == v.size() / 3;
        }
        res.geoms.push_back(g);
    }
    for (auto& l : col.lods) {
        auto* plod = dynamic_cast<osg::PagedLOD*>(l.lod);
        if (!plod) continue;
        for (unsigned i = 0; i < plod->getNumFileNames(); ++i)
            if (!plod->getFileName(i).empty()) res.children.push_back(plod->getFileName(i));
    }
    return res;
}

static std::string infoJson(const InfoResult& r) {
    std::ostringstream o;
    o << std::setprecision(10);
    o << "{\"file\": \"" << jsonEscape(r.file) << "\", \"ok\": " << (r.ok ? "true" : "false");
    if (!r.ok) o << ", \"error\": \"" << jsonEscape(r.error) << "\"";
    o << ", \"geometries\": [";
    for (size_t gi = 0; gi < r.geoms.size(); ++gi) {
        const InfoGeom& g = r.geoms[gi];
        o << (gi ? ", " : "") << "{\"has_finer\": " << (g.hasFiner ? "true" : "false")
          << ", \"num_triangles\": " << g.numTriangles << ", \"depth\": " << g.depth;
        if (g.bb.valid())
            o << ", \"min\": [" << g.bb.xMin() << ", " << g.bb.yMin() << ", " << g.bb.zMin() << "], \"max\": ["
              << g.bb.xMax() << ", " << g.bb.yMax() << ", " << g.bb.zMax() << "]";
        o << ", \"has_uv\": " << (g.hasUV ? "true" : "false") << ", \"image\": " << g.image
          << ", \"texture_missing\": " << (g.textureMissing ? "true" : "false") << "}";
    }
    o << "], \"images\": [";
    for (size_t ii = 0; ii < r.images.size(); ++ii) {
        const InfoImage& im = r.images[ii];
        o << (ii ? ", " : "") << "{\"index\": " << ii << ", \"name\": \"" << jsonEscape(im.name) << "\", \"encoding\": \""
          << jsonEscape(im.encoding) << "\", \"format\": \"" << jsonEscape(im.format)
          << "\", \"external\": " << (im.external ? "true" : "false") << ", \"width\": " << im.width
          << ", \"height\": " << im.height << ", \"bytes\": " << im.bytes << ", \"ok\": " << (im.ok ? "true" : "false");
        if (!im.ok) o << ", \"error\": \"" << jsonEscape(im.error) << "\"";
        o << "}";
    }
    o << "], \"children\": " << jsonStringList(r.children) << "}";
    return o.str();
}

static int cmdInfo(const fs::path& listFile) {
    g_skipImageDecode = true;
    std::ifstream in(listFile, std::ios::binary);
    if (!in) {
        std::cerr << "cannot read " << listFile << std::endl;
        return 2;
    }
    // Exactly one record per input line, also when a file cannot be read.
    std::string path;
    while (std::getline(in, path)) {
        InfoResult res;
        try {
            res = scanFile(path);
        } catch (const std::exception& e) {
            res = InfoResult();
            res.file = path;
            res.error = std::string("exception while reading: ") + e.what();
        } catch (...) {
            res = InfoResult();
            res.file = path;
            res.error = "unknown exception while reading";
        }
        std::cout << infoJson(res) << std::endl;
        g_captured.clear();
        g_missingImageFiles.clear();
        g_failedImages.clear();
    }
    return 0;
}

// ---------------------------------------------------------------------------
// version / selftest: runtime diagnostics
// ---------------------------------------------------------------------------

static const char* kBridgeVersion = "2";

static std::string pluginsJson() {
    osgDB::Registry* reg = osgDB::Registry::instance();
    const char* exts[] = {"osgb", "serializers_osg", "jpg", "png"};
    std::ostringstream o;
    o << "{";
    bool first = true;
    for (const char* e : exts) {
        std::string lib = reg->createLibraryNameForExtension(e);
        std::string found = osgDB::findLibraryFile(lib);
        o << (first ? "" : ", ") << "\"" << e << "\": {\"library\": \"" << jsonEscape(lib) << "\", \"path\": "
          << (found.empty() ? std::string("null") : "\"" + jsonEscape(found) + "\"") << "}";
        first = false;
    }
    o << "}";
    return o.str();
}

static std::string runtimeJsonFields() {
    std::vector<std::string> paths(osgDB::Registry::instance()->getLibraryFilePathList().begin(),
                                   osgDB::Registry::instance()->getLibraryFilePathList().end());
    return std::string("\"osg_version\": \"") + jsonEscape(osgGetVersion()) + "\", \"bridge_version\": \"" + kBridgeVersion +
           "\", \"plugin_paths\": " + jsonStringList(paths) + ", \"plugins\": " + pluginsJson();
}

static int cmdVersion() {
    std::cout << "{" << runtimeJsonFields() << "}" << std::endl;
    return 0;
}

static const int kTestSize = 32;

// Four solid quadrants; output row 0 is the top of the image (v == 1).
static void testColor(int x, int yTop, uint8_t rgb[3]) {
    static const uint8_t colors[4][3] = {{220, 30, 30}, {30, 190, 60}, {30, 60, 220}, {235, 235, 235}};
    int q = (yTop < kTestSize / 2 ? 0 : 2) + (x < kTestSize / 2 ? 0 : 1);
    for (int k = 0; k < 3; ++k) rgb[k] = colors[q][k];
}

static osg::ref_ptr<osg::Image> makeTestImage() {
    osg::ref_ptr<osg::Image> img = new osg::Image;
    img->allocateImage(kTestSize, kTestSize, 1, GL_RGB, GL_UNSIGNED_BYTE);
    img->setInternalTextureFormat(GL_RGB);
    for (int y = 0; y < kTestSize; ++y)
        for (int x = 0; x < kTestSize; ++x) testColor(x, kTestSize - 1 - y, img->data(x, y));
    return img;
}

// Compare top-down RGB(A) pixels with the test pattern.  "margin" skips
// pixels close to the quadrant edges (lossy codecs blur them).
static bool comparePattern(const std::vector<uint8_t>& px, int w, int h, int ch, int margin, int tol, int& worst,
                           std::string& why) {
    worst = 0;
    if (w != kTestSize || h != kTestSize || (ch != 3 && ch != 4) || px.size() != size_t(w) * h * ch) {
        why = "unexpected image size " + std::to_string(w) + "x" + std::to_string(h) + "x" + std::to_string(ch);
        return false;
    }
    const int half = kTestSize / 2;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            int qx = x % half, qy = y % half;
            if (qx < margin || qx >= half - margin || qy < margin || qy >= half - margin) continue;
            uint8_t e[3];
            testColor(x, y, e);
            for (int k = 0; k < 3; ++k) worst = std::max(worst, std::abs(int(px[(size_t(y) * w + x) * ch + k]) - int(e[k])));
        }
    if (worst > tol) {
        why = "max pixel difference " + std::to_string(worst) + " > " + std::to_string(tol) +
              " (wrong orientation, channel order or decoder)";
        return false;
    }
    return true;
}

static bool compareImage(const osg::Image* img, int margin, int tol, int& worst, std::string& why) {
    if (!imageHasData(img)) {
        why = "no pixel data";
        return false;
    }
    std::vector<uint8_t> px;
    int w, h, ch;
    imageToRGB(img, px, w, h, ch);
    return comparePattern(px, w, h, ch, margin, tol, worst, why);
}

static osg::ref_ptr<osg::Geometry> makeTestQuad(const osg::Vec3& offset, osg::Image* img) {
    osg::ref_ptr<osg::Vec3Array> v = new osg::Vec3Array;
    osg::ref_ptr<osg::Vec2Array> uv = new osg::Vec2Array;
    const float corners[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    for (auto& c : corners) {
        v->push_back(offset + osg::Vec3(c[0], c[1], 0.25f * c[0]));
        uv->push_back(osg::Vec2(c[0], c[1]));
    }
    std::vector<uint32_t> tris = {0, 1, 2, 0, 2, 3};
    osg::ref_ptr<osg::Geometry> g = new osg::Geometry;
    g->setUseVertexBufferObjects(true);
    replaceGeometry(*g, v.get(), uv.get(), nullptr, nullptr, tris);
    osg::ref_ptr<osg::Texture2D> tex = new osg::Texture2D(img);
    g->getOrCreateStateSet()->setTextureAttributeAndModes(0, tex.get(), osg::StateAttribute::ON);
    return g;
}

template <typename T>
static bool rawEquals(const fs::path& p, const std::vector<T>& expected) {
    std::vector<T> got;
    return readRaw(p, got) && got == expected;
}

static std::vector<float> quadVertices(const osg::Vec3& offset) {
    std::vector<float> out;
    const float corners[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    for (auto& c : corners) {
        osg::Vec3 p = offset + osg::Vec3(c[0], c[1], 0.25f * c[0]);
        out.insert(out.end(), {p.x(), p.y(), p.z()});
    }
    return out;
}

static int cmdSelftest(const fs::path& dir) {
    const char* names[] = {"jpeg_write", "jpeg_read", "png_write", "png_read", "osgb_write",
                           "osgb_read",  "pixels_match", "geometry_match", "info_scan"};
    std::map<std::string, bool> checks;
    for (const char* n : names) checks[n] = false;
    std::vector<std::string> errors;
    int jpegDiff = -1, pngDiff = -1;
    // keep every image alive so that g_captured keys are never reused
    std::vector<osg::ref_ptr<osg::Referenced>> keep;
    const std::string fineName = "selftest_fine.osgb";
    const osg::Vec3 offA(1000.5f, -20.25f, 3.125f), offB(1002.5f, -20.25f, 3.125f);

    try {
        std::error_code ec;
        fs::create_directories(dir, ec);
        osg::ref_ptr<osg::Image> src = makeTestImage();
        keep.push_back(src.get());

        // --- image plugins --------------------------------------------------
        osg::ref_ptr<osg::Image> images[2];
        const char* exts[2] = {"jpg", "png"};
        const char* labels[2] = {"jpeg", "png"};
        for (int k = 0; k < 2; ++k) {
            fs::path p = dir / (std::string("selftest.") + exts[k]);
            fs::remove(p, ec);
            std::string label = labels[k];
            bool wrote = osgDB::writeImageFile(*src, p.string()) && fs::is_regular_file(p, ec) && fs::file_size(p, ec) > 0;
            checks[label + "_write"] = wrote;
            if (!wrote) {
                errors.push_back("cannot write " + p.string() + " with osgDB::writeImageFile (" + pluginHint(exts[k]) + ")");
                continue;
            }
            osg::ref_ptr<osg::Image> back = osgDB::readRefImageFile(p.string());
            keep.push_back(back.get());
            std::string why;
            int worst = -1;
            // JPEG is lossy; PNG may still pass through colour management
            // (macOS ImageIO), so allow a small difference there too.
            bool ok = back.valid() && compareImage(back.get(), k == 0 ? 3 : 0, k == 0 ? 48 : 12, worst, why);
            (k == 0 ? jpegDiff : pngDiff) = worst;
            checks[label + "_read"] = ok;
            if (!ok) {
                errors.push_back("cannot read back " + p.string() + ": " + (back.valid() ? why : "reader returned no image") +
                                 " (" + pluginHint(exts[k]) + ")");
                continue;
            }
            back->setFileName(std::string("selftest.") + exts[k]);
            images[k] = back;
        }
        for (int k = 0; k < 2; ++k) {
            if (images[k].valid()) continue;
            // fall back to an in-memory copy so that the .osgb checks still run
            images[k] = new osg::Image(*src, osg::CopyOp::DEEP_COPY_ALL);
            images[k]->setFileName(std::string("selftest_raw") + std::to_string(k));
            keep.push_back(images[k].get());
        }

        // --- .osgb write ----------------------------------------------------
        osg::ref_ptr<osg::PagedLOD> plod = new osg::PagedLOD;
        plod->setCenterMode(osg::LOD::USER_DEFINED_CENTER);
        plod->setCenter(osg::Vec3d(1001.5, -19.75, 3.25));
        plod->setRadius(2.0);
        plod->setRangeMode(osg::LOD::PIXEL_SIZE_ON_SCREEN);
        osg::ref_ptr<osg::Geode> geode = new osg::Geode;
        geode->addDrawable(makeTestQuad(offA, images[0].get()).get());
        geode->addDrawable(makeTestQuad(offB, images[1].get()).get());
        plod->addChild(geode.get(), 0.0f, 100.0f);
        plod->setFileName(1, fineName);
        plod->setRange(1, 100.0f, 1e30f);
        fs::path osgbPath = dir / "selftest.osgb";
        fs::remove(osgbPath, ec);
        checks["osgb_write"] = writeNodeEmbedded(plod.get(), osgbPath.string(), {}) && fs::is_regular_file(osgbPath, ec);
        if (!checks["osgb_write"]) errors.push_back("cannot write " + osgbPath.string() + " (" + pluginHint("osgb") + ")");
        g_captured.clear();

        // --- .osgb read -----------------------------------------------------
        osg::ref_ptr<osg::Node> back;
        bool finerOk = false;
        if (checks["osgb_write"]) {
            std::string err;
            back = readNodeChecked(osgbPath.string(), err);
            if (!back) {
                errors.push_back(err);
            } else {
                Collector col;
                back->accept(col);
                for (osg::Image* img : col.images) keep.push_back(img);
                bool lodOk = col.lods.size() == 1 && col.lods[0].paged &&
                             static_cast<osg::PagedLOD*>(col.lods[0].lod)->getNumFileNames() > 1 &&
                             static_cast<osg::PagedLOD*>(col.lods[0].lod)->getFileName(1) == fineName;
                checks["osgb_read"] = lodOk && col.geoms.size() == 2 && col.images.size() == 2;
                finerOk = col.geoms.size() == 2 && col.geoms[0].hasFiner && col.geoms[1].hasFiner;
                if (!checks["osgb_read"])
                    errors.push_back("read back " + osgbPath.string() + " has an unexpected structure (" +
                                     std::to_string(col.geoms.size()) + " geometries, " +
                                     std::to_string(col.images.size()) + " images, " + std::to_string(col.lods.size()) +
                                     " LOD nodes)");
                else if (!finerOk)
                    errors.push_back("PagedLOD child file not recognised as a finer level");
            }
        } else {
            errors.push_back("skipped .osgb read / export checks (write failed)");
        }

        // --- export: decoded pixels and vertices ------------------------------
        if (checks["osgb_read"]) {
            fs::path ex = dir / "export";
            fs::remove_all(ex, ec);
            int rc = cmdExport(osgbPath.string(), ex);
            if (rc != 0) {
                errors.push_back("export failed with code " + std::to_string(rc));
            } else {
                bool pixOk = true;
                for (int k = 0; k < 2; ++k) {
                    fs::path raw = ex / ("tex" + std::to_string(k) + ".raw");
                    std::vector<uint8_t> px;
                    std::string why;
                    int worst = -1;
                    bool ok = readRaw(raw, px) && !px.empty();
                    int ch = ok ? int(px.size() / (kTestSize * kTestSize)) : 0;
                    ok = ok && comparePattern(px, kTestSize, kTestSize, ch, k == 0 ? 3 : 0, k == 0 ? 48 : 12, worst, why);
                    if (!ok) {
                        pixOk = false;
                        errors.push_back("exported texture " + std::to_string(k) + " does not match: " +
                                         (why.empty() ? "missing " + raw.string() : why));
                    }
                }
                checks["pixels_match"] = pixOk;
                std::vector<float> uv = {0, 0, 1, 0, 1, 1, 0, 1};
                std::vector<uint32_t> tris = {0, 1, 2, 0, 2, 3};
                bool geoOk = finerOk;
                const osg::Vec3 offs[2] = {offA, offB};
                for (int k = 0; k < 2; ++k) {
                    std::string base = "g" + std::to_string(k);
                    bool ok = rawEquals(ex / (base + "_v.f32"), quadVertices(offs[k])) &&
                              rawEquals(ex / (base + "_uv.f32"), uv) && rawEquals(ex / (base + "_t.u32"), tris);
                    if (!ok) errors.push_back("exported geometry " + std::to_string(k) + " does not match");
                    geoOk = geoOk && ok;
                }
                checks["geometry_match"] = geoOk;
            }
            g_captured.clear();

            // --- info scan (header probing without decoding) ------------------
            g_skipImageDecode = true;
            InfoResult info = scanFile(osgbPath.string());
            g_skipImageDecode = false;
            bool ok = info.ok && info.geoms.size() == 2 && info.images.size() == 2 && info.children.size() == 1 &&
                      info.children[0] == fineName;
            for (size_t k = 0; ok && k < 2; ++k) {
                const InfoGeom& g = info.geoms[k];
                const InfoImage& im = info.images[k];
                ok = g.hasUV && g.image == int(k) && !g.textureMissing && g.hasFiner && g.numTriangles == 2 && im.ok &&
                     im.width == kTestSize && im.height == kTestSize && im.format == exts[k];
            }
            checks["info_scan"] = ok;
            if (!ok) errors.push_back("info scan of " + osgbPath.string() + " does not match: " + infoJson(info));
            g_captured.clear();
            g_missingImageFiles.clear();
            g_failedImages.clear();
        }
    } catch (const std::exception& e) {
        errors.push_back(std::string("exception: ") + e.what());
    }

    bool allOk = true;
    for (const char* n : names) allOk = allOk && checks[n];
    std::cout << "{\"ok\": " << (allOk ? "true" : "false") << ", " << runtimeJsonFields() << ", \"checks\": {";
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i)
        std::cout << (i ? ", " : "") << "\"" << names[i] << "\": " << (checks[names[i]] ? "true" : "false");
    std::cout << "}, \"pixel_diff\": {\"jpeg\": " << jpegDiff << ", \"png\": " << pngDiff << "}, \"errors\": "
              << jsonStringList(errors) << ", \"dir\": \"" << jsonEscape(dir.string()) << "\"}" << std::endl;
    return allOk ? 0 : 1;
}

int main(int argc, char** argv) {
    osg::setNotifyHandler(new StderrNotifyHandler);
    registerCapturingReaders();
    std::string cmd = argc > 1 ? argv[1] : "";
    try {
        if (cmd == "export" && argc == 4) return cmdExport(argv[2], argv[3]);
        if (cmd == "patch" && argc == 5) return cmdPatch(argv[2], argv[3], argv[4]);
        if (cmd == "build" && argc == 4) return cmdBuild(argv[2], argv[3]);
        if (cmd == "info" && argc == 3) return cmdInfo(argv[2]);
        if (cmd == "version" && argc == 2) return cmdVersion();
        if (cmd == "selftest" && argc == 3) return cmdSelftest(argv[2]);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 10;
    } catch (...) {
        std::cerr << "error: unknown exception" << std::endl;
        return 10;
    }
    std::cerr << "usage:\n"
                 "  osgb_bridge export <in.osgb> <out_dir>\n"
                 "  osgb_bridge patch <in.osgb> <patch_dir> <out.osgb>\n"
                 "  osgb_bridge build <build.txt> <out.osgb>\n"
                 "  osgb_bridge info <file_list.txt>\n"
                 "  osgb_bridge version\n"
                 "  osgb_bridge selftest <tmp_dir>\n";
    return 1;
}
