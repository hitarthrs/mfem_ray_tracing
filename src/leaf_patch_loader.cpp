// Loader for the bilinear leaf-patch JSON exported by
// python_experiments/multiple_step_degree_reduction_surfaces (d4_leaf_bboxes.json).
//
// The files are machine-generated with a fixed shape, so a small recursive-descent
// JSON parser is used instead of pulling in an external dependency.

#include "embree/leaf_patch_loader.hpp"

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
    scene.max_error = root.At("max_error").AsNumber();
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

} // namespace mfem_raytracing
