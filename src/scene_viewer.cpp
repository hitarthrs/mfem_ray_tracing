// Native SDL2 viewer for Embree bilinear patch scene JSONs.
//
// Usage:
//   scene_viewer <*_embree_scene.json | *_leaf_bboxes.json>
//                [--check] [--allow-diagnostic-shell]
//
// Controls:
//   left drag       orbit (orbit mode)
//   left click      fire ray through cursor (shoot mode — toggle with F)
//   mouse wheel     zoom
//   right click     fire an Embree ray through the cursor (either mode)
//   f               toggle orbit / shoot mode (HTML-style click-to-fire)
//   r               recast the grid from the current view
//   c               clear shot rays
//   w/s/h/n         toggle wire/surface/hits/normals
//   1/2/3           top/front/iso views
//   esc/q           quit

#include "embree/leaf_patch_loader.hpp"
#include "embree/raytracer.hpp"

#include <SDL.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

struct Vec3
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
Vec3 operator/(Vec3 a, double s) { return {a.x / s, a.y / s, a.z / s}; }

double Dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 Cross(Vec3 a, Vec3 b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double Len(Vec3 v) { return std::sqrt(Dot(v, v)); }
Vec3 Norm(Vec3 v)
{
    const double len = Len(v);
    return len > 0.0 ? v / len : Vec3{};
}

struct Hit
{
    double t = 0.0;
    int prim = -1;
    Vec3 n;
};

struct Ray
{
    Vec3 o;
    Vec3 d;
    double t_end = 1.0;
    std::vector<Hit> hits;
    bool shot = false;
};

struct Patch
{
    Vec3 p[4]; // P00, P10, P11, P01 for drawing.
};

struct Scene
{
    std::string surface = "surface";
    Vec3 bbox_min;
    Vec3 bbox_max;
    std::vector<Patch> patches;
    std::vector<Ray> rays;
};

class Json
{
public:
    enum class Type
    {
        Null,
        Number,
        String,
        Array,
        Object,
        Bool
    };

    Type type = Type::Null;
    double number = 0.0;
    bool boolean = false;
    std::string string;
    std::vector<Json> array;
    std::map<std::string, Json> object;

    bool Has(const std::string &key) const { return object.find(key) != object.end(); }
    const Json &At(const std::string &key) const
    {
        const auto it = object.find(key);
        if (it == object.end())
        {
            throw std::runtime_error("missing JSON key '" + key + "'");
        }
        return it->second;
    }
};

class JsonParser
{
public:
    explicit JsonParser(std::string text) : text_(std::move(text)) {}

    Json Parse()
    {
        Json v = ParseValue();
        SkipWs();
        if (pos_ != text_.size())
        {
            throw std::runtime_error("trailing data in JSON");
        }
        return v;
    }

private:
    Json ParseValue()
    {
        SkipWs();
        if (pos_ >= text_.size())
        {
            throw std::runtime_error("unexpected end of JSON");
        }
        const char c = text_[pos_];
        if (c == '{') return ParseObject();
        if (c == '[') return ParseArray();
        if (c == '"') return ParseString();
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return ParseNumber();
        if (Match("true"))
        {
            Json v;
            v.type = Json::Type::Bool;
            v.boolean = true;
            return v;
        }
        if (Match("false"))
        {
            Json v;
            v.type = Json::Type::Bool;
            v.boolean = false;
            return v;
        }
        if (Match("null")) return Json{};
        throw std::runtime_error("invalid JSON value");
    }

    Json ParseObject()
    {
        Json out;
        out.type = Json::Type::Object;
        Expect('{');
        SkipWs();
        if (Peek('}'))
        {
            ++pos_;
            return out;
        }
        while (true)
        {
            Json key = ParseString();
            SkipWs();
            Expect(':');
            out.object.emplace(key.string, ParseValue());
            SkipWs();
            if (Peek('}'))
            {
                ++pos_;
                break;
            }
            Expect(',');
        }
        return out;
    }

    Json ParseArray()
    {
        Json out;
        out.type = Json::Type::Array;
        Expect('[');
        SkipWs();
        if (Peek(']'))
        {
            ++pos_;
            return out;
        }
        while (true)
        {
            out.array.push_back(ParseValue());
            SkipWs();
            if (Peek(']'))
            {
                ++pos_;
                break;
            }
            Expect(',');
        }
        return out;
    }

    Json ParseString()
    {
        Json out;
        out.type = Json::Type::String;
        Expect('"');
        while (pos_ < text_.size())
        {
            char c = text_[pos_++];
            if (c == '"') return out;
            if (c == '\\')
            {
                if (pos_ >= text_.size()) throw std::runtime_error("bad JSON escape");
                c = text_[pos_++];
                if (c == '"' || c == '\\' || c == '/') out.string.push_back(c);
                else if (c == 'b') out.string.push_back('\b');
                else if (c == 'f') out.string.push_back('\f');
                else if (c == 'n') out.string.push_back('\n');
                else if (c == 'r') out.string.push_back('\r');
                else if (c == 't') out.string.push_back('\t');
                else if (c == 'u')
                {
                    pos_ = std::min(pos_ + 4, text_.size());
                    out.string.push_back('?');
                }
                else
                {
                    throw std::runtime_error("bad JSON escape");
                }
            }
            else
            {
                out.string.push_back(c);
            }
        }
        throw std::runtime_error("unterminated JSON string");
    }

    Json ParseNumber()
    {
        const std::size_t start = pos_;
        if (Peek('-')) ++pos_;
        while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        if (Peek('.'))
        {
            ++pos_;
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        }
        if (Peek('e') || Peek('E'))
        {
            ++pos_;
            if (Peek('+') || Peek('-')) ++pos_;
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        }
        Json out;
        out.type = Json::Type::Number;
        out.number = std::stod(text_.substr(start, pos_ - start));
        return out;
    }

    bool Match(const char *word)
    {
        const std::size_t n = std::char_traits<char>::length(word);
        if (text_.compare(pos_, n, word) != 0) return false;
        pos_ += n;
        return true;
    }

    bool Peek(char c) const { return pos_ < text_.size() && text_[pos_] == c; }

    void Expect(char c)
    {
        SkipWs();
        if (!Peek(c))
        {
            throw std::runtime_error(std::string("expected '") + c + "'");
        }
        ++pos_;
    }

    void SkipWs()
    {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    }

    std::string text_;
    std::size_t pos_ = 0;
};

double Num(const Json &j)
{
    if (j.type != Json::Type::Number) throw std::runtime_error("expected number");
    return j.number;
}

Vec3 Vec(const Json &j)
{
    if (j.type != Json::Type::Array || j.array.size() != 3) throw std::runtime_error("expected vec3");
    return {Num(j.array[0]), Num(j.array[1]), Num(j.array[2])};
}

std::string Basename(const std::string &path)
{
    const std::size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

std::string ReadFile(const std::string &path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open '" + path + "'");
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

Scene LoadSceneJson(const std::string &path, bool allow_diagnostic_shell = false)
{
    const Json root = JsonParser(ReadFile(path)).Parse();
    if (root.type != Json::Type::Object) throw std::runtime_error("top-level JSON is not an object");

    // scene_viewer also accepts leaf JSON directly, rather than going through
    // LoadLeafPatchScene, so preserve the same explicit safety gate here.
    if (root.Has("certification"))
    {
        const Json &certification = root.At("certification");
        if (certification.type != Json::Type::Object || !certification.Has("rt_certified") ||
            certification.At("rt_certified").type != Json::Type::Bool)
        {
            throw std::runtime_error("invalid baked-shell certification metadata");
        }
        if (!certification.At("rt_certified").boolean && !allow_diagnostic_shell)
        {
            throw std::runtime_error(
                "scene declares rt_certified=false; pass --allow-diagnostic-shell to inspect it");
        }
    }

    Scene scene;
    if (root.Has("surface")) scene.surface = root.At("surface").string;

    if (root.Has("patches"))
    {
        const Json &patches = root.At("patches");
        for (const Json &p : patches.array)
        {
            if (p.type != Json::Type::Array || p.array.size() != 4) continue;
            Patch patch;
            for (int i = 0; i < 4; ++i) patch.p[i] = Vec(p.array[i]);
            scene.patches.push_back(patch);
        }
    }
    else if (root.Has("leaves"))
    {
        const Json &leaves = root.At("leaves");
        for (const Json &leaf : leaves.array)
        {
            const Json &cp = leaf.At("control_points");
            if (cp.array.size() != 2 || cp.array[0].array.size() != 2 || cp.array[1].array.size() != 2)
            {
                continue;
            }
            Patch patch;
            patch.p[0] = Vec(cp.array[0].array[0]);
            patch.p[1] = Vec(cp.array[1].array[0]);
            patch.p[2] = Vec(cp.array[1].array[1]);
            patch.p[3] = Vec(cp.array[0].array[1]);
            scene.patches.push_back(patch);
        }
    }

    if (root.Has("rays"))
    {
        for (const Json &r : root.At("rays").array)
        {
            Ray ray;
            ray.o = Vec(r.At("o"));
            ray.d = Norm(Vec(r.At("d")));
            ray.t_end = r.Has("tEnd") ? Num(r.At("tEnd")) : 1.0;
            if (r.Has("hits"))
            {
                for (const Json &h : r.At("hits").array)
                {
                    Hit hit;
                    hit.t = Num(h.At("t"));
                    hit.prim = h.Has("prim") ? static_cast<int>(Num(h.At("prim"))) : -1;
                    hit.n = h.Has("n") ? Vec(h.At("n")) : Vec3{};
                    ray.hits.push_back(hit);
                }
            }
            else if (r.Has("tHit") && r.At("tHit").type == Json::Type::Number)
            {
                Hit hit;
                hit.t = Num(r.At("tHit"));
                hit.prim = r.Has("prim") ? static_cast<int>(Num(r.At("prim"))) : -1;
                hit.n = r.Has("n") ? Vec(r.At("n")) : Vec3{};
                ray.hits.push_back(hit);
            }
            scene.rays.push_back(ray);
        }
    }

    if (root.Has("bbox_min") && root.Has("bbox_max"))
    {
        scene.bbox_min = Vec(root.At("bbox_min"));
        scene.bbox_max = Vec(root.At("bbox_max"));
    }
    else if (!scene.patches.empty())
    {
        scene.bbox_min = scene.bbox_max = scene.patches[0].p[0];
        for (const Patch &p : scene.patches)
        {
            for (Vec3 q : p.p)
            {
                scene.bbox_min.x = std::min(scene.bbox_min.x, q.x);
                scene.bbox_min.y = std::min(scene.bbox_min.y, q.y);
                scene.bbox_min.z = std::min(scene.bbox_min.z, q.z);
                scene.bbox_max.x = std::max(scene.bbox_max.x, q.x);
                scene.bbox_max.y = std::max(scene.bbox_max.y, q.y);
                scene.bbox_max.z = std::max(scene.bbox_max.z, q.z);
            }
        }
    }

    if (scene.patches.empty()) throw std::runtime_error("no patches found in JSON");
    const double min_t_end = std::max(1.0, Len(scene.bbox_max - scene.bbox_min)) * 4.0;
    for (Ray &ray : scene.rays)
    {
        ray.t_end = std::max(ray.t_end, min_t_end);
    }
    return scene;
}

mfem_raytracing::BilinearPatchPrimitive ToEmbreePatch(const Patch &patch)
{
    mfem_raytracing::BilinearPatchPrimitive out;
    const int map[4] = {
        static_cast<int>(mfem_raytracing::BilinearCorner::P00),
        static_cast<int>(mfem_raytracing::BilinearCorner::P10),
        static_cast<int>(mfem_raytracing::BilinearCorner::P11),
        static_cast<int>(mfem_raytracing::BilinearCorner::P01),
    };
    for (int i = 0; i < 4; ++i)
    {
        out.control_points[map[i]][0] = patch.p[i].x;
        out.control_points[map[i]][1] = patch.p[i].y;
        out.control_points[map[i]][2] = patch.p[i].z;
    }
    return out;
}

std::unique_ptr<mfem_raytracing::EmbreeRayTracer> BuildTracer(const std::vector<Patch> &patches)
{
    auto tracer = std::make_unique<mfem_raytracing::EmbreeRayTracer>();
    std::vector<mfem_raytracing::BilinearPatchPrimitive> embree_patches;
    embree_patches.reserve(patches.size());
    for (const Patch &patch : patches) embree_patches.push_back(ToEmbreePatch(patch));
    tracer->RegisterPatches(std::move(embree_patches));
    tracer->CommitScene();
    return tracer;
}

struct Camera
{
    double yaw = -0.75;
    double pitch = 0.55;
    double zoom = 1.0;
    Vec3 center;
    double radius = 1.0;
};

struct ScreenPoint
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Basis
{
    Vec3 right;
    Vec3 up;
    Vec3 forward;
};

/// Eye distance along +forward from scene center (matches Project / unproject).
/// Must be shared by CastShot origin and RayDirectionFromScreen — a mismatch
/// makes the ray miss the clicked surface (common "1 HIT / weird aim" bug).
constexpr double kEyeDistanceFactor = 3.0;

Basis CameraBasis(const Camera &cam)
{
    const double cp = std::cos(cam.pitch);
    Vec3 forward{std::sin(cam.yaw) * cp, std::cos(cam.yaw) * cp, std::sin(cam.pitch)};
    forward = Norm(forward);
    Vec3 right = Norm(Cross(forward, {0, 0, 1}));
    Vec3 up = Norm(Cross(right, forward));
    return {right, up, forward};
}

Vec3 CameraEye(const Camera &cam)
{
    const Basis b = CameraBasis(cam);
    return cam.center + b.forward * (cam.radius * kEyeDistanceFactor / cam.zoom);
}

ScreenPoint Project(Vec3 p, const Camera &cam, int w, int h)
{
    const Basis b = CameraBasis(cam);
    const Vec3 v = p - cam.center;
    const double scale = 0.42 * std::min(w, h) * cam.zoom / std::max(1e-9, cam.radius);
    return {w * 0.5 + Dot(v, b.right) * scale,
            h * 0.52 - Dot(v, b.up) * scale,
            Dot(v, b.forward)};
}

/// Perspective ray through the clicked pixel, same idea as bilinear_ray_tracer.html:
/// origin = camera eye, direction = toward the unprojected point on the view plane
/// through the scene center.
Vec3 ScreenPointOnViewPlane(int mx, int my, const Camera &cam, int w, int h)
{
    const Basis b = CameraBasis(cam);
    const double scale = 0.42 * std::min(w, h) * cam.zoom / std::max(1e-9, cam.radius);
    const double x = (mx - w * 0.5) / scale;
    const double y = -(my - h * 0.52) / scale;
    return cam.center + b.right * x + b.up * y;
}

Vec3 RayDirectionFromScreen(int mx, int my, const Camera &cam, int w, int h)
{
    return Norm(ScreenPointOnViewPlane(mx, my, cam, w, h) - CameraEye(cam));
}

void SetColor(SDL_Renderer *r, int red, int green, int blue, int alpha = 255)
{
    SDL_SetRenderDrawColor(r, static_cast<Uint8>(red), static_cast<Uint8>(green),
                           static_cast<Uint8>(blue), static_cast<Uint8>(alpha));
}

void Line(SDL_Renderer *r, ScreenPoint a, ScreenPoint b)
{
    SDL_RenderDrawLineF(r, static_cast<float>(a.x), static_cast<float>(a.y),
                        static_cast<float>(b.x), static_cast<float>(b.y));
}

void DrawText(SDL_Renderer *r, int x, int y, const std::string &text, int scale = 2)
{
    static const std::map<char, std::array<unsigned char, 7>> font = {
        {' ', {0, 0, 0, 0, 0, 0, 0}}, {'-', {0, 0, 0, 31, 0, 0, 0}},
        {'.', {0, 0, 0, 0, 0, 12, 12}}, {'/', {1, 2, 4, 8, 16, 0, 0}},
        {'0', {14, 17, 19, 21, 25, 17, 14}}, {'1', {4, 12, 4, 4, 4, 4, 14}},
        {'2', {14, 17, 1, 2, 4, 8, 31}}, {'3', {30, 1, 1, 14, 1, 1, 30}},
        {'4', {2, 6, 10, 18, 31, 2, 2}}, {'5', {31, 16, 30, 1, 1, 17, 14}},
        {'6', {6, 8, 16, 30, 17, 17, 14}}, {'7', {31, 1, 2, 4, 8, 8, 8}},
        {'8', {14, 17, 17, 14, 17, 17, 14}}, {'9', {14, 17, 17, 15, 1, 2, 12}},
        {':', {0, 12, 12, 0, 12, 12, 0}}, {'_', {0, 0, 0, 0, 0, 0, 31}},
        {'+', {0, 4, 4, 31, 4, 4, 0}}, {'*', {0, 10, 4, 31, 4, 10, 0}},
        {'A', {14, 17, 17, 31, 17, 17, 17}}, {'B', {30, 17, 17, 30, 17, 17, 30}},
        {'C', {14, 17, 16, 16, 16, 17, 14}}, {'D', {30, 17, 17, 17, 17, 17, 30}},
        {'E', {31, 16, 16, 30, 16, 16, 31}}, {'F', {31, 16, 16, 30, 16, 16, 16}},
        {'G', {14, 17, 16, 23, 17, 17, 15}}, {'H', {17, 17, 17, 31, 17, 17, 17}},
        {'I', {14, 4, 4, 4, 4, 4, 14}}, {'J', {7, 2, 2, 2, 2, 18, 12}},
        {'K', {17, 18, 20, 24, 20, 18, 17}}, {'L', {16, 16, 16, 16, 16, 16, 31}},
        {'M', {17, 27, 21, 21, 17, 17, 17}}, {'N', {17, 25, 21, 19, 17, 17, 17}},
        {'O', {14, 17, 17, 17, 17, 17, 14}}, {'P', {30, 17, 17, 30, 16, 16, 16}},
        {'Q', {14, 17, 17, 17, 21, 18, 13}}, {'R', {30, 17, 17, 30, 20, 18, 17}},
        {'S', {15, 16, 16, 14, 1, 1, 30}}, {'T', {31, 4, 4, 4, 4, 4, 4}},
        {'U', {17, 17, 17, 17, 17, 17, 14}}, {'V', {17, 17, 17, 17, 10, 10, 4}},
        {'W', {17, 17, 17, 21, 21, 27, 17}}, {'X', {17, 17, 10, 4, 10, 17, 17}},
        {'Y', {17, 17, 10, 4, 4, 4, 4}}, {'Z', {31, 1, 2, 4, 8, 16, 31}},
    };
    int cx = x;
    for (char raw : text)
    {
        const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(raw)));
        const auto it = font.find(c);
        const auto glyph = it == font.end() ? font.at(' ') : it->second;
        for (int row = 0; row < 7; ++row)
        {
            for (int col = 0; col < 5; ++col)
            {
                if (glyph[row] & (1 << (4 - col)))
                {
                    SDL_Rect px{cx + col * scale, y + row * scale, scale, scale};
                    SDL_RenderFillRect(r, &px);
                }
            }
        }
        cx += 6 * scale;
    }
}

int TextWidth(const std::string &text, int scale = 2)
{
    return static_cast<int>(text.size()) * 6 * scale;
}

void FillRect(SDL_Renderer *r, SDL_Rect rect, int red, int green, int blue, int alpha)
{
    SetColor(r, red, green, blue, alpha);
    SDL_RenderFillRect(r, &rect);
}

void StrokeRect(SDL_Renderer *r, SDL_Rect rect, int red, int green, int blue, int alpha)
{
    SetColor(r, red, green, blue, alpha);
    SDL_RenderDrawRect(r, &rect);
}

// Soft panel: fill + hairline stroke + left accent bar.
void DrawPanel(SDL_Renderer *r, SDL_Rect rect, bool accent = true)
{
    FillRect(r, rect, 243, 248, 245, 228);
    StrokeRect(r, rect, 30, 54, 42, 55);
    FillRect(r, SDL_Rect{rect.x + 1, rect.y + 1, rect.w - 2, 1}, 255, 255, 255, 70);
    if (accent)
    {
        FillRect(r, SDL_Rect{rect.x, rect.y, 4, rect.h}, 11, 122, 110, 230);
    }
}

void DrawDivider(SDL_Renderer *r, int x, int y, int w)
{
    SetColor(r, 30, 54, 42, 36);
    SDL_RenderDrawLine(r, x, y, x + w, y);
    SetColor(r, 255, 255, 255, 55);
    SDL_RenderDrawLine(r, x, y + 1, x + w, y + 1);
}

void DrawMetric(SDL_Renderer *r, int x, int y, const std::string &label, const std::string &value,
                bool emphasize = false)
{
    SetColor(r, 95, 118, 108, 230);
    DrawText(r, x, y, label, 1);
    if (emphasize)
    {
        SetColor(r, 212, 85, 26, 245);
    }
    else
    {
        SetColor(r, 16, 32, 24, 245);
    }
    DrawText(r, x + 112, y, value, 2);
}

void DrawPill(SDL_Renderer *r, SDL_Rect pill, const std::string &label, bool enabled)
{
    if (enabled)
    {
        FillRect(r, pill, 11, 122, 110, 48);
        StrokeRect(r, pill, 11, 122, 110, 170);
        SetColor(r, 8, 78, 70, 245);
    }
    else
    {
        FillRect(r, pill, 255, 255, 255, 55);
        StrokeRect(r, pill, 40, 62, 52, 50);
        SetColor(r, 110, 128, 118, 210);
    }
    const int tw = TextWidth(label, 1);
    DrawText(r, pill.x + (pill.w - tw) / 2, pill.y + (pill.h - 7) / 2, label, 1);
}

std::string CountHits(const std::vector<Ray> &rays)
{
    std::size_t total = 0;
    for (const Ray &ray : rays) total += ray.hits.size();
    return std::to_string(total);
}

struct UiLayout
{
    SDL_Rect dock;
    SDL_Rect surf;
    SDL_Rect wire;
    SDL_Rect hits;
    SDL_Rect normals;
    SDL_Rect recast_grid;
    SDL_Rect clear_grid;
    SDL_Rect clear_shots;
};

UiLayout MakeUiLayout(int /*w*/, int h)
{
    const int left = 28;
    const int top = 282;
    const int gap = 8;
    const int pill_w = 68;
    const int pill_h = 24;
    return {
        SDL_Rect{18, 18, 320, std::min(540, h - 140)},
        SDL_Rect{left, top, pill_w, pill_h},
        SDL_Rect{left + pill_w + gap, top, pill_w, pill_h},
        SDL_Rect{left + 2 * (pill_w + gap), top, pill_w, pill_h},
        SDL_Rect{left + 3 * (pill_w + gap), top, pill_w, pill_h},
        SDL_Rect{left, top + 48, 88, 26},
        SDL_Rect{left + 96, top + 48, 88, 26},
        SDL_Rect{left + 192, top + 48, 88, 26},
    };
}

bool Contains(SDL_Rect rect, int x, int y)
{
    return x >= rect.x && x < rect.x + rect.w && y >= rect.y && y < rect.y + rect.h;
}

void DrawButton(SDL_Renderer *r, SDL_Rect rect, const std::string &label, bool primary = false)
{
    if (primary)
    {
        FillRect(r, rect, 11, 122, 110, 55);
        StrokeRect(r, rect, 11, 122, 110, 190);
        SetColor(r, 8, 70, 64, 245);
    }
    else
    {
        FillRect(r, rect, 255, 255, 255, 72);
        StrokeRect(r, rect, 40, 62, 52, 70);
        SetColor(r, 28, 44, 36, 235);
    }
    const int tw = TextWidth(label, 1);
    DrawText(r, rect.x + (rect.w - tw) / 2, rect.y + (rect.h - 7) / 2, label, 1);
}

void DrawDot(SDL_Renderer *r, double x, double y, float radius, int red, int green, int blue, int alpha)
{
    SetColor(r, red, green, blue, alpha);
    const int steps = 10;
    for (int i = 0; i < steps; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(steps - 1);
        const float rr = radius * (1.0f - 0.35f * t);
        SDL_FRect rect{static_cast<float>(x) - rr, static_cast<float>(y) - rr, rr * 2.0f, rr * 2.0f};
        SDL_RenderFillRectF(r, &rect);
    }
}

void DrawOverlay(SDL_Renderer *r, const Scene &scene, const std::vector<Ray> &shots,
                 bool surface, bool wire, bool hits, bool normals, bool shoot_mode, int w, int h,
                 const std::string &current_path)
{
    SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_BLEND);
    const UiLayout ui = MakeUiLayout(w, h);

    // ---- left dock ----
    DrawPanel(r, ui.dock, true);
    SetColor(r, 11, 122, 110, 245);
    DrawText(r, 36, 34, "SCENE VIEWER", 2);
    SetColor(r, 95, 118, 108, 220);
    DrawText(r, 36, 56, "EMBREE  *  BILINEAR", 1);
    SetColor(r, 16, 32, 24, 240);
    DrawText(r, 36, 74, scene.surface.substr(0, 40), 1);
    SetColor(r, 95, 118, 108, 210);
    {
        std::string file_line = "FILE " + Basename(current_path);
        if (file_line.size() > 40) file_line = file_line.substr(0, 37) + "...";
        DrawText(r, 36, 92, file_line, 1);
    }
    DrawDivider(r, 36, 112, ui.dock.w - 40);

    DrawMetric(r, 36, 128, "PATCHES", std::to_string(scene.patches.size()));
    DrawMetric(r, 36, 154, "GRID RAYS", std::to_string(scene.rays.size()));
    DrawMetric(r, 36, 180, "GRID HITS", CountHits(scene.rays), true);
    DrawMetric(r, 36, 206, "SHOTS", std::to_string(shots.size()));
    DrawMetric(r, 36, 232, "SHOT HITS", CountHits(shots), true);

    DrawDivider(r, 36, ui.surf.y - 24, ui.dock.w - 40);
    SetColor(r, 95, 118, 108, 230);
    DrawText(r, 36, ui.surf.y - 16, "LAYERS", 1);
    DrawPill(r, ui.surf, "SURF", surface);
    DrawPill(r, ui.wire, "WIRE", wire);
    DrawPill(r, ui.hits, "HITS", hits);
    DrawPill(r, ui.normals, "NORM", normals);

    SetColor(r, 95, 118, 108, 230);
    DrawText(r, 36, ui.recast_grid.y - 18, "ACTIONS", 1);
    DrawButton(r, ui.recast_grid, "RECAST", true);
    DrawButton(r, ui.clear_grid, "CLR GRID");
    DrawButton(r, ui.clear_shots, "CLR SHOT");

    SetColor(r, 70, 92, 82, 210);
    DrawText(r, 36, ui.recast_grid.y + 40, "LMB ORBIT   WHEEL ZOOM", 1);
    DrawText(r, 36, ui.recast_grid.y + 56, "RMB SHOOT   R GRID  C SHOTS", 1);
    SetColor(r, 11, 122, 110, 220);
    DrawText(r, 36, ui.recast_grid.y + 76, "DROP JSON FILE TO LOAD", 1);

    // ---- right readout ----
    const int hud_w = 236;
    SDL_Rect hud{w - hud_w - 18, 18, hud_w, shots.empty() ? 118 : 168};
    DrawPanel(r, hud, true);
    SetColor(r, 95, 118, 108, 230);
    DrawText(r, hud.x + 16, 34, "READOUT", 1);
    FillRect(r, SDL_Rect{hud.x + hud.w - 26, 36, 8, 8}, 26, 163, 146, 230);
    SetColor(r, 11, 122, 110, 245);
    DrawText(r, hud.x + 16, 56, "LIVE EMBREE TRACE", 1);
    SetColor(r, 28, 44, 36, 230);
    if (shoot_mode)
    {
        DrawText(r, hud.x + 16, 78, "SHOOT MODE  F=ORBIT", 1);
        DrawText(r, hud.x + 16, 96, "LMB FIRES THROUGH CURSOR", 1);
    }
    else
    {
        DrawText(r, hud.x + 16, 78, "ORBIT MODE  F=SHOOT", 1);
        DrawText(r, hud.x + 16, 96, "RMB / SHIFT+LMB FIRE", 1);
    }

    if (!shots.empty())
    {
        const Ray &last = shots.back();
        DrawDivider(r, hud.x + 16, 116, hud.w - 36);
        SetColor(r, 95, 118, 108, 230);
        DrawText(r, hud.x + 16, 128, "LAST SHOT", 1);
        if (last.hits.empty())
        {
            SetColor(r, 110, 128, 118, 230);
            DrawText(r, hud.x + 16, 148, "MISS", 2);
        }
        else
        {
            SetColor(r, 212, 85, 26, 245);
            DrawText(r, hud.x + 16, 148,
                     std::to_string(last.hits.size()) + " HIT" + (last.hits.size() == 1 ? "" : "S"), 2);
            std::ostringstream ts;
            ts << std::fixed << std::setprecision(2);
            for (std::size_t i = 0; i < last.hits.size() && i < 3; ++i)
            {
                if (i) ts << " ";
                ts << "T" << last.hits[i].t;
            }
            if (last.hits.size() > 3) ts << " +";
            SetColor(r, 28, 44, 36, 220);
            DrawText(r, hud.x + 16, 170, ts.str(), 1);
        }
    }
}

// Top-center banner: a persistent error (load failed) takes priority over the
// fade-out "loaded" toast, matching the html viewer's inline load feedback.
void DrawStatusBanner(SDL_Renderer *r, const std::string &error_message,
                     const std::string &loaded_path, int toast_alpha, int w)
{
    if (!error_message.empty())
    {
        std::string msg = "LOAD FAILED: " + error_message;
        if (msg.size() > 64) msg = msg.substr(0, 61) + "...";
        const int banner_w = std::min(600, std::max(260, TextWidth(msg, 1) + 32));
        SDL_Rect banner{(w - banner_w) / 2, 18, banner_w, 34};
        FillRect(r, banner, 224, 84, 64, 235);
        StrokeRect(r, banner, 150, 40, 30, 255);
        SetColor(r, 255, 245, 240, 250);
        DrawText(r, banner.x + 16, banner.y + (banner.h - 7) / 2, msg, 1);
        return;
    }
    if (toast_alpha <= 0) return;
    const std::string msg = "LOADED " + Basename(loaded_path);
    const int banner_w = std::min(600, std::max(220, TextWidth(msg, 1) + 32));
    SDL_Rect banner{(w - banner_w) / 2, 18, banner_w, 34};
    FillRect(r, banner, 26, 163, 146, toast_alpha);
    StrokeRect(r, banner, 11, 122, 110, std::min(255, toast_alpha + 20));
    SetColor(r, 8, 40, 34, toast_alpha);
    DrawText(r, banner.x + 16, banner.y + (banner.h - 7) / 2, msg, 1);
}

void DrawAxisGizmo(SDL_Renderer *r, const Camera &cam, int w, int h)
{
    const Basis b = CameraBasis(cam);
    const int ox = w - 78;
    const int oy = h - 64;
    auto project_axis = [&](Vec3 axis) {
        return ScreenPoint{static_cast<double>(ox) + Dot(axis, b.right) * 36.0,
                           static_cast<double>(oy) - Dot(axis, b.up) * 36.0,
                           0.0};
    };
    SDL_Rect box{w - 128, h - 116, 110, 98};
    DrawPanel(r, box, false);
    SetColor(r, 95, 118, 108, 220);
    DrawText(r, box.x + 12, box.y + 10, "AXES", 1);
    SetColor(r, 200, 72, 58, 235);
    Line(r, {static_cast<double>(ox), static_cast<double>(oy), 0.0}, project_axis({1, 0, 0}));
    DrawText(r, static_cast<int>(project_axis({1, 0, 0}).x) + 4,
             static_cast<int>(project_axis({1, 0, 0}).y) - 4, "X", 1);
    SetColor(r, 32, 140, 88, 235);
    Line(r, {static_cast<double>(ox), static_cast<double>(oy), 0.0}, project_axis({0, 1, 0}));
    DrawText(r, static_cast<int>(project_axis({0, 1, 0}).x) + 4,
             static_cast<int>(project_axis({0, 1, 0}).y) - 4, "Y", 1);
    SetColor(r, 48, 98, 176, 235);
    Line(r, {static_cast<double>(ox), static_cast<double>(oy), 0.0}, project_axis({0, 0, 1}));
    DrawText(r, static_cast<int>(project_axis({0, 0, 1}).x) + 4,
             static_cast<int>(project_axis({0, 0, 1}).y) - 4, "Z", 1);
}

SDL_Vertex Vertex(ScreenPoint p, SDL_Color color)
{
    SDL_Vertex v;
    v.position.x = static_cast<float>(p.x);
    v.position.y = static_cast<float>(p.y);
    v.color = color;
    v.tex_coord.x = 0.0f;
    v.tex_coord.y = 0.0f;
    return v;
}

void DrawRay(SDL_Renderer *renderer, const Ray &ray, const Camera &cam, int w, int h)
{
    Vec3 prev = ray.o;
    if (ray.hits.empty())
    {
        SetColor(renderer, 125, 140, 132, ray.shot ? 200 : 80);
        Line(renderer, Project(ray.o, cam, w, h), Project(ray.o + ray.d * ray.t_end, cam, w, h));
        return;
    }

    // Incoming to first hit.
    SetColor(renderer, ray.shot ? 212 : 11, ray.shot ? 85 : 122, ray.shot ? 26 : 110, ray.shot ? 235 : 130);
    for (std::size_t i = 0; i < ray.hits.size(); ++i)
    {
        if (i == 1)
        {
            // Between / after hits: quieter pass-through.
            SetColor(renderer, ray.shot ? 212 : 11, ray.shot ? 85 : 122, ray.shot ? 26 : 110,
                     ray.shot ? 120 : 55);
        }
        Vec3 p = ray.o + ray.d * ray.hits[i].t;
        Line(renderer, Project(prev, cam, w, h), Project(p, cam, w, h));
        prev = p;
    }
    SetColor(renderer, 11, 122, 110, ray.shot ? 100 : 45);
    Line(renderer, Project(prev, cam, w, h), Project(ray.o + ray.d * ray.t_end, cam, w, h));
}

void DrawScene(SDL_Renderer *renderer, const Scene &scene, const std::vector<Ray> &shots,
               const Camera &cam, int w, int h, bool show_surface, bool show_wire,
               bool show_hits, bool show_normals, bool shoot_mode,
               const std::string &current_path,
               const std::string &error_message, int toast_alpha)
{
    // Sage field + soft vignette grid.
    SetColor(renderer, 232, 240, 235);
    SDL_RenderClear(renderer);
    SetColor(renderer, 30, 54, 42, 22);
    for (int x = 0; x < w; x += 48) SDL_RenderDrawLine(renderer, x, 0, x, h);
    for (int y = 0; y < h; y += 48) SDL_RenderDrawLine(renderer, 0, y, w, y);
    FillRect(renderer, SDL_Rect{0, 0, w, 90}, 255, 255, 255, 28);
    FillRect(renderer, SDL_Rect{0, h - 110, w, 110}, 16, 40, 30, 18);

    struct DepthPatch
    {
        double z;
        const Patch *patch;
    };
    std::vector<DepthPatch> order;
    order.reserve(scene.patches.size());
    for (const Patch &patch : scene.patches)
    {
        double z = 0.0;
        for (Vec3 p : patch.p) z += Project(p, cam, w, h).z;
        order.push_back({z / 4.0, &patch});
    }
    std::sort(order.begin(), order.end(), [](const DepthPatch &a, const DepthPatch &b) { return a.z < b.z; });

    for (const DepthPatch &dp : order)
    {
        const Patch &patch = *dp.patch;
        std::array<ScreenPoint, 4> p{};
        for (int i = 0; i < 4; ++i) p[i] = Project(patch.p[i], cam, w, h);

        if (show_surface)
        {
            const double nz = std::abs(Dot(Norm(Cross(patch.p[1] - patch.p[0], patch.p[3] - patch.p[0])),
                                           CameraBasis(cam).forward));
            const Uint8 g = static_cast<Uint8>(145 + 70 * nz);
            const Uint8 rb = static_cast<Uint8>(g > 12 ? g - 12 : 0);
            SDL_Color fill{rb, g, static_cast<Uint8>(std::min(255, g + 6)), 178};
            SDL_Vertex verts[4] = {Vertex(p[0], fill), Vertex(p[1], fill), Vertex(p[2], fill), Vertex(p[3], fill)};
            const int idx[6] = {0, 1, 2, 0, 2, 3};
            SDL_RenderGeometry(renderer, nullptr, verts, 4, idx, 6);
        }
        if (show_wire)
        {
            SetColor(renderer, 40, 62, 52, 95);
            Line(renderer, p[0], p[1]);
            Line(renderer, p[1], p[2]);
            Line(renderer, p[2], p[3]);
            Line(renderer, p[3], p[0]);
        }
    }

    for (const Ray &ray : scene.rays) DrawRay(renderer, ray, cam, w, h);
    for (const Ray &ray : shots) DrawRay(renderer, ray, cam, w, h);

    if (show_hits)
    {
        auto draw_hits = [&](const std::vector<Ray> &rays, bool shot) {
            for (const Ray &ray : rays)
            {
                for (const Hit &hit : ray.hits)
                {
                    ScreenPoint sp = Project(ray.o + ray.d * hit.t, cam, w, h);
                    if (shot)
                    {
                        DrawDot(renderer, sp.x, sp.y, 5.0f, 212, 85, 26, 235);
                    }
                    else
                    {
                        DrawDot(renderer, sp.x, sp.y, 3.5f, 232, 150, 70, 210);
                    }
                    if (show_normals)
                    {
                        SetColor(renderer, 48, 98, 176, 150);
                        Vec3 hp = ray.o + ray.d * hit.t;
                        Line(renderer, Project(hp, cam, w, h),
                             Project(hp + Norm(hit.n) * (cam.radius * 0.045), cam, w, h));
                    }
                }
            }
        };
        draw_hits(scene.rays, false);
        draw_hits(shots, true);
    }

    DrawAxisGizmo(renderer, cam, w, h);
    DrawOverlay(renderer, scene, shots, show_surface, show_wire, show_hits, show_normals,
                shoot_mode, w, h, current_path);
    DrawStatusBanner(renderer, error_message, current_path, toast_alpha, w);
    SDL_RenderPresent(renderer);
}

std::vector<Ray> CastGrid(mfem_raytracing::EmbreeRayTracer &tracer, const Camera &cam, int w, int h)
{
    std::vector<Ray> rays;
    const int n = 15;
    const Basis b = CameraBasis(cam);
    const Vec3 origin_base = cam.center + b.forward * (cam.radius * 5.0 / cam.zoom);
    const double span = cam.radius * 1.8 / cam.zoom;
    const double t_end = cam.radius * 14.0 / cam.zoom;
    for (int iy = 0; iy < n; ++iy)
    {
        for (int ix = 0; ix < n; ++ix)
        {
            const double fx = (ix + 0.5) / n - 0.5;
            const double fy = (iy + 0.5) / n - 0.5;
            Ray ray;
            ray.o = origin_base + b.right * (fx * span) + b.up * (fy * span);
            ray.d = b.forward * -1.0;
            ray.t_end = t_end;
            double o[3] = {ray.o.x, ray.o.y, ray.o.z};
            double d[3] = {ray.d.x, ray.d.y, ray.d.z};
            for (const auto &h : tracer.IntersectAll(o, d, 0.0, t_end))
            {
                ray.hits.push_back({h.t, static_cast<int>(h.prim_id), {h.Ng[0], h.Ng[1], h.Ng[2]}});
            }
            rays.push_back(ray);
        }
    }
    return rays;
}

Ray CastShot(mfem_raytracing::EmbreeRayTracer &tracer, const Camera &cam, int w, int h, int mx, int my)
{
    Ray ray;
    // Same eye used by RayDirectionFromScreen (was 5× vs 3× — click aimed wrong).
    ray.o = CameraEye(cam);
    ray.d = RayDirectionFromScreen(mx, my, cam, w, h);
    // Clear the far side of the scene from the eye (mirrors bilinear HTML sceneTEnd).
    const double eye_dist = cam.radius * kEyeDistanceFactor / cam.zoom;
    ray.t_end = eye_dist + cam.radius * 4.0;
    ray.shot = true;
    double o[3] = {ray.o.x, ray.o.y, ray.o.z};
    double d[3] = {ray.d.x, ray.d.y, ray.d.z};
    for (const auto &h : tracer.IntersectAll(o, d, 0.0, ray.t_end))
    {
        ray.hits.push_back({h.t, static_cast<int>(h.prim_id), {h.Ng[0], h.Ng[1], h.Ng[2]}});
    }
    return ray;
}

void FitCamera(Camera &cam, const Scene &scene)
{
    cam.center = (scene.bbox_min + scene.bbox_max) * 0.5;
    cam.radius = std::max(1e-6, Len(scene.bbox_max - scene.bbox_min) * 0.5);
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: scene_viewer <*_embree_scene.json | *_leaf_bboxes.json>"
                  << " [--check] [--allow-diagnostic-shell]\n";
        return 1;
    }
    bool check_only = false;
    bool allow_diagnostic_shell = false;
    for (int i = 2; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--check") { check_only = true; }
        else if (arg == "--allow-diagnostic-shell") { allow_diagnostic_shell = true; }
        else
        {
            std::cerr << "error: unknown argument '" << arg << "'\n";
            return 1;
        }
    }

    Scene scene;
    try
    {
        scene = LoadSceneJson(argv[1], allow_diagnostic_shell);
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    if (check_only)
    {
        auto tracer = BuildTracer(scene.patches);
        std::cout << "scene_viewer check ok\n"
                  << "  file:    " << argv[1] << "\n"
                  << "  surface: " << scene.surface << "\n"
                  << "  patches: " << scene.patches.size() << "\n"
                  << "  rays:    " << scene.rays.size() << "\n";
        return 0;
    }

    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << "\n";
        return 1;
    }

    SDL_Window *window = SDL_CreateWindow(("Scene Viewer · " + scene.surface).c_str(),
                                          SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                          1280, 860, SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    if (!window)
    {
        std::cerr << "SDL_CreateWindow failed: " << SDL_GetError() << "\n";
        SDL_Quit();
        return 1;
    }
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!renderer)
    {
        std::cerr << "SDL_CreateRenderer failed: " << SDL_GetError() << "\n";
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    SDL_EventState(SDL_DROPFILE, SDL_ENABLE);

    auto tracer = BuildTracer(scene.patches);

    Camera cam;
    FitCamera(cam, scene);
    std::vector<Ray> shots;
    bool show_surface = true;
    bool show_wire = true;
    bool show_hits = true;
    bool show_normals = false;
    bool shoot_mode = false;  // F toggles; left-click fires like bilinear HTML
    bool dragging = false;
    int last_x = 0;
    int last_y = 0;
    bool running = true;

    std::string current_path = argv[1];
    std::string error_message;
    Uint32 last_loaded_ticks = SDL_GetTicks();

    // Reload the scene from `path` in place: on success, swaps scene/tracer,
    // refits the camera and clears shots; on failure, leaves everything as-is
    // and surfaces the error banner instead of crashing (mirrors the html
    // viewer's drag-and-drop behaviour).
    auto try_load = [&](const std::string &path) {
        Scene new_scene;
        try
        {
            new_scene = LoadSceneJson(path, allow_diagnostic_shell);
        }
        catch (const std::exception &e)
        {
            error_message = e.what();
            std::cerr << "error loading '" << path << "': " << e.what() << "\n";
            return;
        }
        auto new_tracer = BuildTracer(new_scene.patches);
        scene = std::move(new_scene);
        tracer = std::move(new_tracer);
        shots.clear();
        FitCamera(cam, scene);
        current_path = path;
        error_message.clear();
        last_loaded_ticks = SDL_GetTicks();
        SDL_SetWindowTitle(window, ("Scene Viewer \xc2\xb7 " + scene.surface).c_str());
        std::cout << "scene_viewer loaded '" << path << "'\n"
                  << "  patches: " << scene.patches.size() << "\n"
                  << "  rays:    " << scene.rays.size() << "\n";
    };

    auto render_mouse = [&](int x, int y) {
        int win_w = 1;
        int win_h = 1;
        int out_w = 1;
        int out_h = 1;
        SDL_GetWindowSize(window, &win_w, &win_h);
        SDL_GetRendererOutputSize(renderer, &out_w, &out_h);
        return std::pair<int, int>{static_cast<int>(std::lround(x * (static_cast<double>(out_w) / win_w))),
                                   static_cast<int>(std::lround(y * (static_cast<double>(out_h) / win_h)))};
    };

    auto handle_ui_click = [&](int x, int y) {
        const auto [rx, ry] = render_mouse(x, y);
        int out_w = 0;
        int out_h = 0;
        SDL_GetRendererOutputSize(renderer, &out_w, &out_h);
        const UiLayout ui = MakeUiLayout(out_w, out_h);
        if (Contains(ui.surf, rx, ry))
        {
            show_surface = !show_surface;
            return true;
        }
        if (Contains(ui.wire, rx, ry))
        {
            show_wire = !show_wire;
            return true;
        }
        if (Contains(ui.hits, rx, ry))
        {
            show_hits = !show_hits;
            return true;
        }
        if (Contains(ui.normals, rx, ry))
        {
            show_normals = !show_normals;
            return true;
        }
        if (Contains(ui.recast_grid, rx, ry))
        {
            scene.rays = CastGrid(*tracer, cam, out_w, out_h);
            return true;
        }
        if (Contains(ui.clear_grid, rx, ry))
        {
            scene.rays.clear();
            return true;
        }
        if (Contains(ui.clear_shots, rx, ry))
        {
            shots.clear();
            return true;
        }
        return false;
    };

    std::cout << "scene_viewer loaded '" << argv[1] << "'\n"
              << "  patches: " << scene.patches.size() << "\n"
              << "  rays:    " << scene.rays.size() << "\n"
              << "controls: left-drag orbit, F shoot-mode (LMB fire), right-click / shift+LMB fire,\n"
              << "          wheel zoom, r recast, x clear grid, c clear shots, w/s/h/n toggles\n"
              << "          drag & drop a leaf-bbox / scene JSON onto the window to reload\n";

    while (running)
    {
        SDL_Event ev;
        while (SDL_PollEvent(&ev))
        {
            if (ev.type == SDL_QUIT) running = false;
            if (ev.type == SDL_DROPFILE)
            {
                if (char *dropped = ev.drop.file)
                {
                    try_load(dropped);
                    SDL_free(dropped);
                }
            }
            if (ev.type == SDL_MOUSEBUTTONDOWN && ev.button.button == SDL_BUTTON_LEFT)
            {
                if (!handle_ui_click(ev.button.x, ev.button.y))
                {
                    const bool shift =
                        (SDL_GetModState() & (KMOD_LSHIFT | KMOD_RSHIFT)) != 0;
                    if (shoot_mode || shift)
                    {
                        int w = 0;
                        int h = 0;
                        SDL_GetRendererOutputSize(renderer, &w, &h);
                        const auto [rx, ry] = render_mouse(ev.button.x, ev.button.y);
                        shots.push_back(CastShot(*tracer, cam, w, h, rx, ry));
                    }
                    else
                    {
                        dragging = true;
                        last_x = ev.button.x;
                        last_y = ev.button.y;
                    }
                }
            }
            if (ev.type == SDL_MOUSEBUTTONUP && ev.button.button == SDL_BUTTON_LEFT) dragging = false;
            if (ev.type == SDL_MOUSEMOTION && dragging)
            {
                const int dx = ev.motion.x - last_x;
                const int dy = ev.motion.y - last_y;
                cam.yaw += dx * 0.008;
                cam.pitch = std::max(-1.45, std::min(1.45, cam.pitch + dy * 0.008));
                last_x = ev.motion.x;
                last_y = ev.motion.y;
            }
            if (ev.type == SDL_MOUSEWHEEL)
            {
                cam.zoom *= std::pow(1.12, ev.wheel.y);
                cam.zoom = std::max(0.08, std::min(80.0, cam.zoom));
            }
            if (ev.type == SDL_MOUSEBUTTONDOWN && ev.button.button == SDL_BUTTON_RIGHT)
            {
                int w = 0;
                int h = 0;
                SDL_GetRendererOutputSize(renderer, &w, &h);
                const auto [rx, ry] = render_mouse(ev.button.x, ev.button.y);
                shots.push_back(CastShot(*tracer, cam, w, h, rx, ry));
            }
            if (ev.type == SDL_KEYDOWN)
            {
                switch (ev.key.keysym.sym)
                {
                case SDLK_ESCAPE:
                case SDLK_q:
                    running = false;
                    break;
                case SDLK_f:
                    shoot_mode = !shoot_mode;
                    dragging = false;
                    break;
                case SDLK_w:
                    show_wire = !show_wire;
                    break;
                case SDLK_s:
                    show_surface = !show_surface;
                    break;
                case SDLK_h:
                    show_hits = !show_hits;
                    break;
                case SDLK_n:
                    show_normals = !show_normals;
                    break;
                case SDLK_c:
                    shots.clear();
                    break;
                case SDLK_x:
                    scene.rays.clear();
                    break;
                case SDLK_r:
                {
                    int w = 0;
                    int h = 0;
                    SDL_GetRendererOutputSize(renderer, &w, &h);
                    scene.rays = CastGrid(*tracer, cam, w, h);
                    break;
                }
                case SDLK_1:
                    cam.yaw = 0.0;
                    cam.pitch = 1.45;
                    break;
                case SDLK_2:
                    cam.yaw = 0.0;
                    cam.pitch = 0.0;
                    break;
                case SDLK_3:
                    cam.yaw = -0.75;
                    cam.pitch = 0.55;
                    break;
                default:
                    break;
                }
            }
        }

        int w = 0;
        int h = 0;
        SDL_GetRendererOutputSize(renderer, &w, &h);

        // Fade the "loaded" toast out over ~0.7s after an initial 0.9s hold.
        int toast_alpha = 0;
        {
            const Uint32 elapsed = SDL_GetTicks() - last_loaded_ticks;
            constexpr Uint32 kHoldMs = 900;
            constexpr Uint32 kFadeMs = 700;
            if (elapsed < kHoldMs)
            {
                toast_alpha = 235;
            }
            else if (elapsed < kHoldMs + kFadeMs)
            {
                const double frac = 1.0 - static_cast<double>(elapsed - kHoldMs) / kFadeMs;
                toast_alpha = static_cast<int>(std::lround(235.0 * frac));
            }
        }

        DrawScene(renderer, scene, shots, cam, w, h, show_surface, show_wire, show_hits,
                 show_normals, shoot_mode, current_path, error_message, toast_alpha);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
