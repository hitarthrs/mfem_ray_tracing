// Loader for the bilinear leaf-patch JSON exported by
// python_experiments/multiple_step_degree_reduction_surfaces (d4_leaf_bboxes.json).
//
// The files are machine-generated with a fixed shape, so a small recursive-descent
// JSON parser is used instead of pulling in an external dependency.

#include "embree/leaf_patch_loader.hpp"
#include "hard_seam_bilinearization.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace mfem_raytracing
{
namespace
{

struct JsonValue
{
    enum class Type
    {
        Null,
        Bool,
        Number,
        String,
        Array,
        Object
    };

    Type type = Type::Null;
    bool bool_value = false;
    double number_value = 0.0;
    std::string string_value;
    std::vector<JsonValue> array_items;
    std::map<std::string, JsonValue> object_items;

    bool IsNull() const { return type == Type::Null; }

    const JsonValue &At(const std::string &key) const
    {
        const auto it = object_items.find(key);
        if (type != Type::Object || it == object_items.end())
        {
            throw std::runtime_error("leaf patch JSON: missing key '" + key + "'");
        }
        return it->second;
    }

    double AsNumber() const
    {
        if (type != Type::Number)
        {
            throw std::runtime_error("leaf patch JSON: expected a number");
        }
        return number_value;
    }

    const std::vector<JsonValue> &AsArray() const
    {
        if (type != Type::Array)
        {
            throw std::runtime_error("leaf patch JSON: expected an array");
        }
        return array_items;
    }

    const std::string &AsString() const
    {
        if (type != Type::String)
        {
            throw std::runtime_error("leaf patch JSON: expected a string");
        }
        return string_value;
    }
};

class JsonParser
{
public:
    explicit JsonParser(const std::string &text) : text_(text) {}

    JsonValue Parse()
    {
        JsonValue value = ParseValue();
        SkipWhitespace();
        if (pos_ != text_.size())
        {
            Fail("trailing characters after JSON document");
        }
        return value;
    }

private:
    const std::string &text_;
    std::size_t pos_ = 0;

    [[noreturn]] void Fail(const std::string &message) const
    {
        std::ostringstream oss;
        oss << "leaf patch JSON parse error at offset " << pos_ << ": " << message;
        throw std::runtime_error(oss.str());
    }

    void SkipWhitespace()
    {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_])))
        {
            ++pos_;
        }
    }

    char Peek()
    {
        if (pos_ >= text_.size())
        {
            Fail("unexpected end of input");
        }
        return text_[pos_];
    }

    void Expect(char c)
    {
        if (Peek() != c)
        {
            Fail(std::string("expected '") + c + "'");
        }
        ++pos_;
    }

    bool TryConsume(char c)
    {
        if (pos_ < text_.size() && text_[pos_] == c)
        {
            ++pos_;
            return true;
        }
        return false;
    }

    void ExpectLiteral(const char *literal)
    {
        for (const char *p = literal; *p != '\0'; ++p)
        {
            if (pos_ >= text_.size() || text_[pos_] != *p)
            {
                Fail(std::string("expected literal '") + literal + "'");
            }
            ++pos_;
        }
    }

    JsonValue ParseValue()
    {
        SkipWhitespace();
        const char c = Peek();
        switch (c)
        {
            case '{':
                return ParseObject();
            case '[':
                return ParseArray();
            case '"':
            {
                JsonValue value;
                value.type = JsonValue::Type::String;
                value.string_value = ParseString();
                return value;
            }
            case 't':
            {
                ExpectLiteral("true");
                JsonValue value;
                value.type = JsonValue::Type::Bool;
                value.bool_value = true;
                return value;
            }
            case 'f':
            {
                ExpectLiteral("false");
                JsonValue value;
                value.type = JsonValue::Type::Bool;
                value.bool_value = false;
                return value;
            }
            case 'n':
            {
                ExpectLiteral("null");
                return JsonValue{};
            }
            default:
                return ParseNumber();
        }
    }

    JsonValue ParseObject()
    {
        Expect('{');
        JsonValue value;
        value.type = JsonValue::Type::Object;
        SkipWhitespace();
        if (TryConsume('}'))
        {
            return value;
        }
        while (true)
        {
            SkipWhitespace();
            std::string key = ParseString();
            SkipWhitespace();
            Expect(':');
            value.object_items[std::move(key)] = ParseValue();
            SkipWhitespace();
            if (TryConsume(','))
            {
                continue;
            }
            Expect('}');
            return value;
        }
    }

    JsonValue ParseArray()
    {
        Expect('[');
        JsonValue value;
        value.type = JsonValue::Type::Array;
        SkipWhitespace();
        if (TryConsume(']'))
        {
            return value;
        }
        while (true)
        {
            value.array_items.push_back(ParseValue());
            SkipWhitespace();
            if (TryConsume(','))
            {
                continue;
            }
            Expect(']');
            return value;
        }
    }

    std::string ParseString()
    {
        Expect('"');
        std::string out;
        while (true)
        {
            if (pos_ >= text_.size())
            {
                Fail("unterminated string");
            }
            const char c = text_[pos_++];
            if (c == '"')
            {
                return out;
            }
            if (c == '\\')
            {
                if (pos_ >= text_.size())
                {
                    Fail("unterminated escape sequence");
                }
                const char esc = text_[pos_++];
                switch (esc)
                {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u':
                    {
                        // \uXXXX → UTF-8 (BMP only; enough for our leaf JSON metadata).
                        if (pos_ + 4 > text_.size())
                        {
                            Fail("unterminated unicode escape");
                        }
                        unsigned code = 0;
                        for (int k = 0; k < 4; ++k)
                        {
                            const char h = text_[pos_++];
                            code <<= 4;
                            if (h >= '0' && h <= '9')
                            {
                                code |= static_cast<unsigned>(h - '0');
                            }
                            else if (h >= 'a' && h <= 'f')
                            {
                                code |= static_cast<unsigned>(h - 'a' + 10);
                            }
                            else if (h >= 'A' && h <= 'F')
                            {
                                code |= static_cast<unsigned>(h - 'A' + 10);
                            }
                            else
                            {
                                Fail("invalid unicode escape");
                            }
                        }
                        if (code <= 0x7F)
                        {
                            out.push_back(static_cast<char>(code));
                        }
                        else if (code <= 0x7FF)
                        {
                            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
                            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                        }
                        else
                        {
                            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
                            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                        }
                        break;
                    }
                    default:
                        Fail("unsupported string escape");
                }
            }
            else
            {
                out.push_back(c);
            }
        }
    }

    JsonValue ParseNumber()
    {
        const std::size_t start = pos_;
        if (TryConsume('-'))
        {
        }
        while (pos_ < text_.size() &&
               (std::isdigit(static_cast<unsigned char>(text_[pos_])) || text_[pos_] == '.' ||
                text_[pos_] == 'e' || text_[pos_] == 'E' || text_[pos_] == '+' || text_[pos_] == '-'))
        {
            ++pos_;
        }
        if (pos_ == start)
        {
            Fail("invalid number");
        }
        JsonValue value;
        value.type = JsonValue::Type::Number;
        value.number_value = std::strtod(text_.substr(start, pos_ - start).c_str(), nullptr);
        return value;
    }
};

void ReadVec3(const JsonValue &array, double out[3])
{
    const auto &items = array.AsArray();
    if (items.size() != 3)
    {
        throw std::runtime_error("leaf patch JSON: expected a 3-vector");
    }
    for (int k = 0; k < 3; ++k)
    {
        out[k] = items[k].AsNumber();
    }
}

void ReadDomain(const JsonValue &array, double out[2])
{
    const auto &items = array.AsArray();
    if (items.size() != 2)
    {
        throw std::runtime_error("leaf patch JSON: expected a 2-vector domain");
    }
    out[0] = items[0].AsNumber();
    out[1] = items[1].AsNumber();
}

// JSON control_points/weights use geomdl [i][j] = [u][v] ordering; map the 2x2
// net onto the BilinearCorner layout.
int CornerFromUV(int i, int j)
{
    if (i == 0)
    {
        return static_cast<int>(j == 0 ? BilinearCorner::P00 : BilinearCorner::P01);
    }
    return static_cast<int>(j == 0 ? BilinearCorner::P10 : BilinearCorner::P11);
}

LeafPatch ParseLeaf(const JsonValue &leaf_json)
{
    LeafPatch leaf;
    leaf.index = static_cast<int>(leaf_json.At("index").AsNumber());
    const auto patch_id_it = leaf_json.object_items.find("patch_id");
    if (patch_id_it != leaf_json.object_items.end())
    {
        leaf.patch_id = static_cast<int>(patch_id_it->second.AsNumber());
    }
    const auto role_it = leaf_json.object_items.find("role");
    if (role_it != leaf_json.object_items.end())
    {
        leaf.role = role_it->second.AsString();
    }
    ReadDomain(leaf_json.At("u_domain_global"), leaf.u_domain_global);
    ReadDomain(leaf_json.At("v_domain_global"), leaf.v_domain_global);
    leaf.total_error = leaf_json.At("total_error").AsNumber();
    ReadVec3(leaf_json.At("bbox_min"), leaf.bbox.min);
    ReadVec3(leaf_json.At("bbox_max"), leaf.bbox.max);

    const int degree_u = static_cast<int>(leaf_json.At("degree_u").AsNumber());
    const int degree_v = static_cast<int>(leaf_json.At("degree_v").AsNumber());
    if (degree_u != 1 || degree_v != 1)
    {
        throw std::runtime_error("leaf patch JSON: leaf is not degree (1, 1)");
    }

    const auto &rows = leaf_json.At("control_points").AsArray();
    if (rows.size() != 2 || rows[0].AsArray().size() != 2 || rows[1].AsArray().size() != 2)
    {
        throw std::runtime_error("leaf patch JSON: control_points is not a 2x2 net");
    }
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            ReadVec3(rows[i].AsArray()[j], leaf.patch.control_points[CornerFromUV(i, j)]);
        }
    }

    const JsonValue &weights = leaf_json.At("weights");
    if (weights.IsNull())
    {
        leaf.patch.rational = false;
    }
    else
    {
        const auto &weight_rows = weights.AsArray();
        if (weight_rows.size() != 2 || weight_rows[0].AsArray().size() != 2 ||
            weight_rows[1].AsArray().size() != 2)
        {
            throw std::runtime_error("leaf patch JSON: weights is not a 2x2 net");
        }
        leaf.patch.rational = true;
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                leaf.patch.weights[CornerFromUV(i, j)] = weight_rows[i].AsArray()[j].AsNumber();
            }
        }
    }
    return leaf;
}

} // namespace

std::vector<BilinearPatchPrimitive> LeafPatchScene::Patches() const
{
    std::vector<BilinearPatchPrimitive> patches;
    patches.reserve(leaves.size());
    for (const LeafPatch &leaf : leaves)
    {
        patches.push_back(leaf.patch);
    }
    return patches;
}

LeafPatchScene LoadLeafPatchScene(const std::string &json_path)
{
    std::ifstream stream(json_path);
    if (!stream)
    {
        throw std::runtime_error("leaf patch JSON: cannot open '" + json_path + "'");
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    const std::string text = buffer.str();

    const JsonValue root = JsonParser(text).Parse();

    LeafPatchScene scene;
    scene.surface_name = root.At("surface").string_value;
    const auto max_err_it = root.object_items.find("max_error");
    if (max_err_it != root.object_items.end())
    {
        scene.max_error = max_err_it->second.AsNumber();
    }
    else
    {
        // Optional for baked shells that report max_decomp_err instead.
        const auto decomp_it = root.object_items.find("max_decomp_err");
        scene.max_error =
            (decomp_it != root.object_items.end()) ? decomp_it->second.AsNumber() : 0.0;
    }
    ReadVec3(root.At("scene_bbox_min"), scene.scene_bbox.min);
    ReadVec3(root.At("scene_bbox_max"), scene.scene_bbox.max);

    const auto &leaves = root.At("leaves").AsArray();
    scene.leaves.reserve(leaves.size());
    for (const JsonValue &leaf_json : leaves)
    {
        scene.leaves.push_back(ParseLeaf(leaf_json));
    }
    return scene;
}

SurfaceData LoadSurfaceDataJson(const std::string &json_path)
{
    std::ifstream stream(json_path);
    if (!stream)
    {
        throw std::runtime_error("surface JSON: cannot open '" + json_path + "'");
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    const JsonValue root = JsonParser(buffer.str()).Parse();

    SurfaceData surface;
    surface.degree_u = static_cast<int>(root.At("degree_u").AsNumber());
    surface.degree_v = static_cast<int>(root.At("degree_v").AsNumber());
    surface.dim = 3;

    const auto &rows = root.At("control_points").AsArray();
    surface.control_points.resize(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        const auto &cols = rows[i].AsArray();
        surface.control_points[i].resize(cols.size());
        for (std::size_t j = 0; j < cols.size(); ++j)
        {
            const auto &comps = cols[j].AsArray();
            if (comps.size() != 3)
            {
                throw std::runtime_error("surface JSON: expected 3D control points");
            }
            surface.control_points[i][j] = {comps[0].AsNumber(), comps[1].AsNumber(),
                                            comps[2].AsNumber()};
        }
    }

    const JsonValue &weights = root.At("weights");
    if (!weights.IsNull())
    {
        const auto &w_rows = weights.AsArray();
        surface.weights.resize(w_rows.size());
        for (std::size_t i = 0; i < w_rows.size(); ++i)
        {
            const auto &w_cols = w_rows[i].AsArray();
            surface.weights[i].resize(w_cols.size());
            for (std::size_t j = 0; j < w_cols.size(); ++j)
            {
                surface.weights[i][j] = w_cols[j].AsNumber();
            }
        }
    }

    for (const JsonValue &k : root.At("knotvector_u").AsArray())
    {
        surface.knotvector_u.push_back(k.AsNumber());
    }
    for (const JsonValue &k : root.At("knotvector_v").AsArray())
    {
        surface.knotvector_v.push_back(k.AsNumber());
    }

    const auto pu = static_cast<std::size_t>(surface.degree_u);
    const auto pv = static_cast<std::size_t>(surface.degree_v);
    if (surface.knotvector_u.size() <= 2 * pu + 1 || surface.knotvector_v.size() <= 2 * pv + 1)
    {
        throw std::runtime_error("surface JSON: knot vectors too short for the degrees");
    }
    surface.u_domain = {surface.knotvector_u[pu],
                        surface.knotvector_u[surface.knotvector_u.size() - pu - 1]};
    surface.v_domain = {surface.knotvector_v[pv],
                        surface.knotvector_v[surface.knotvector_v.size() - pv - 1]};
    return surface;
}

SurfacePatchCatalog LoadSurfacePatchCatalogJson(const std::string &json_path)
{
    std::ifstream stream(json_path);
    if (!stream)
    {
        throw std::runtime_error("surface patch catalog: cannot open '" + json_path + "'");
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    const JsonValue root = JsonParser(buffer.str()).Parse();

    const auto parse_surface = [](const JsonValue &node) {
        SurfaceData surface;
        surface.degree_u = static_cast<int>(node.At("degree_u").AsNumber());
        surface.degree_v = static_cast<int>(node.At("degree_v").AsNumber());
        surface.dim = 3;
        const auto &rows = node.At("control_points").AsArray();
        surface.control_points.resize(rows.size());
        for (std::size_t i = 0; i < rows.size(); ++i)
        {
            const auto &cols = rows[i].AsArray();
            surface.control_points[i].resize(cols.size());
            for (std::size_t j = 0; j < cols.size(); ++j)
            {
                const auto &point = cols[j].AsArray();
                if (point.size() != 3)
                {
                    throw std::runtime_error("surface patch catalog: expected 3D control points");
                }
                surface.control_points[i][j] = {point[0].AsNumber(), point[1].AsNumber(),
                                                point[2].AsNumber()};
            }
        }
        const JsonValue &weights = node.At("weights");
        if (!weights.IsNull())
        {
            const auto &rows_w = weights.AsArray();
            surface.weights.resize(rows_w.size());
            for (std::size_t i = 0; i < rows_w.size(); ++i)
            {
                const auto &cols_w = rows_w[i].AsArray();
                surface.weights[i].resize(cols_w.size());
                for (std::size_t j = 0; j < cols_w.size(); ++j)
                {
                    surface.weights[i][j] = cols_w[j].AsNumber();
                }
            }
        }
        for (const JsonValue &k : node.At("knotvector_u").AsArray())
        {
            surface.knotvector_u.push_back(k.AsNumber());
        }
        for (const JsonValue &k : node.At("knotvector_v").AsArray())
        {
            surface.knotvector_v.push_back(k.AsNumber());
        }
        const std::size_t pu = static_cast<std::size_t>(surface.degree_u);
        const std::size_t pv = static_cast<std::size_t>(surface.degree_v);
        if (surface.knotvector_u.size() <= 2 * pu + 1 ||
            surface.knotvector_v.size() <= 2 * pv + 1)
        {
            throw std::runtime_error("surface patch catalog: knot vectors too short for the degrees");
        }
        surface.u_domain = {surface.knotvector_u[pu],
                            surface.knotvector_u[surface.knotvector_u.size() - pu - 1]};
        surface.v_domain = {surface.knotvector_v[pv],
                            surface.knotvector_v[surface.knotvector_v.size() - pv - 1]};
        return surface;
    };

    SurfacePatchCatalog catalog;
    const auto mesh_it = root.object_items.find("mesh");
    if (mesh_it != root.object_items.end()) { catalog.mesh = mesh_it->second.AsString(); }
    const auto description_it = root.object_items.find("description");
    if (description_it != root.object_items.end()) { catalog.description = description_it->second.AsString(); }
    const auto &patches = root.At("patches").AsArray();
    catalog.patches.reserve(patches.size());
    for (const JsonValue &node : patches)
    {
        SurfacePatchDescriptor patch;
        patch.id = static_cast<int>(node.At("id").AsNumber());
        patch.name = node.At("name").AsString();
        patch.role = node.At("role").AsString();
        const auto quarter_it = node.object_items.find("quarter");
        if (quarter_it != node.object_items.end()) { patch.quarter = static_cast<int>(quarter_it->second.AsNumber()); }
        const auto volume_it = node.object_items.find("volume_patch");
        if (volume_it != node.object_items.end()) { patch.volume_patch = static_cast<int>(volume_it->second.AsNumber()); }
        const auto attribute_it = node.object_items.find("attribute");
        if (attribute_it != node.object_items.end()) { patch.attribute = static_cast<int>(attribute_it->second.AsNumber()); }
        patch.surface = parse_surface(node);
        // geomdl (used by the reference Python pipeline) normalizes its
        // parameter domain. Preserve the geometry while adopting [0,1]^2 so
        // hard knots, leaf domains, and exported JSON are directly comparable.
        const auto normalize_knots = [](std::vector<double> &knots, int degree,
                                        std::pair<double, double> &domain) {
            const double lo = knots[static_cast<std::size_t>(degree)];
            const double hi = knots[knots.size() - static_cast<std::size_t>(degree) - 1];
            if (!(hi > lo))
            {
                throw std::runtime_error("surface patch catalog: invalid parameter domain");
            }
            for (double &knot : knots)
            {
                knot = (knot - lo) / (hi - lo);
            }
            domain = {0.0, 1.0};
        };
        normalize_knots(patch.surface.knotvector_u, patch.surface.degree_u, patch.surface.u_domain);
        normalize_knots(patch.surface.knotvector_v, patch.surface.degree_v, patch.surface.v_domain);
        catalog.patches.push_back(std::move(patch));
    }
    return catalog;
}

const SurfacePatchDescriptor &FindSurfacePatch(const SurfacePatchCatalog &catalog, int id)
{
    for (const SurfacePatchDescriptor &patch : catalog.patches)
    {
        if (patch.id == id)
        {
            return patch;
        }
    }
    throw std::out_of_range("surface patch catalog: no patch with id " + std::to_string(id));
}

} // namespace mfem_raytracing
