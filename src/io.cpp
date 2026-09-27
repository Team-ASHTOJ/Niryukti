#include "json.hpp"
#include "vantage/vantage.hpp"
#include <fstream>
#include <iomanip>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
namespace vantage {
using json = nlohmann::json;
namespace {
std::string trim(std::string s) {
    auto a = s.find_first_not_of(" \t\r\n");
    return a == s.npos ? "" : s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}
std::string upper(std::string s) {
    for (auto &c : s)
        c = std::toupper(static_cast<unsigned char>(c));
    return s;
}
std::vector<std::string> words(const std::string &s) {
    std::istringstream in(s);
    std::vector<std::string> t;
    std::string a;
    while (in >> a)
        t.push_back(a);
    return t;
}
double number(std::string s) {
    for (auto &c : s)
        if (c == 'D' || c == 'd')
            c = 'e';
    size_t end;
    double v = std::stod(s, &end);
    if (end != s.size() || !std::isfinite(v))
        throw std::runtime_error("Invalid finite number: " + s);
    return v;
}
double bound(const json &j, double fallback) {
    if (j.is_null())
        return fallback;
    if (j.is_string()) {
        auto s = upper(j.get<std::string>());
        if (s == "INF" || s == "+INF")
            return inf;
        if (s == "-INF")
            return -inf;
        throw std::runtime_error("Invalid bound string");
    }
    return j.get<double>();
}
int64_t addvar(Model &m, const std::string &s) {
    auto n = m.c.size();
    m.names.push_back(s);
    m.c.push_back(0);
    m.q.push_back(0);
    m.lb.push_back(0);
    m.ub.push_back(inf);
    m.types.push_back(VarType::Continuous);
    return n;
}
void sense(Model &m, std::string s) {
    s = upper(s);
    if (s == "MIN" || s == "MINIMIZE")
        m.sense = 1;
    else if (s == "MAX" || s == "MAXIMIZE")
        m.sense = -1;
    else
        throw std::runtime_error("Unknown objective sense");
}
void canonicalize(Model &m) {
    for (auto &v : m.c)
        v *= m.sense;
    for (auto &v : m.q)
        v *= m.sense;
    for (auto &v : m.Q.value)
        v *= m.sense;
    m.offset *= m.sense;
    m.validate();
}
Model read_json(std::istream &in) {
    json j;
    in >> j;
    Model m;
    m.name = j.value("name", "model");
    sense(m, j.value("sense", "min"));
    std::map<std::string, int64_t> ids;
    for (const auto &v : j.at("variables")) {
        auto s = v.at("name").get<std::string>();
        if (ids.count(s))
            throw std::runtime_error("Duplicate variable");
        auto k = addvar(m, s);
        ids[s] = k;
        m.lb[k] = bound(v.value("lb", json(0)), -inf);
        m.ub[k] = bound(v.value("ub", json()), inf);
        auto t = v.value("type", "continuous");
        if (t == "integer")
            m.types[k] = VarType::Integer;
        else if (t == "binary") {
            m.types[k] = VarType::Binary;
            m.lb[k] = std::max(0., m.lb[k]);
            m.ub[k] = std::min(1., m.ub[k]);
        } else if (t != "continuous")
            throw std::runtime_error("Unknown variable type");
    }
    const auto &obj = j.at("objective");
    m.c = obj.at("linear").get<std::vector<double>>();
    if (obj.contains("quadratic_diagonal"))
        m.q = obj.at("quadratic_diagonal").get<std::vector<double>>();
    if (obj.contains("quadratic_sparse")) {
        std::vector<Entry> entries;
        for (const auto &v : obj.at("quadratic_sparse")) {
            if (v.size() != 3)
                throw std::runtime_error("quadratic_sparse entries must be [row,column,value]");
            entries.push_back({v[0].get<int64_t>(), v[1].get<int64_t>(), v[2].get<double>()});
        }
        m.Q = Sparse::build(m.c.size(), m.c.size(), std::move(entries));
    }
    if (obj.contains("quadratic"))
        throw std::runtime_error("Use explicit symmetric quadratic_sparse triples");
    m.offset = obj.value("offset", 0.);
    std::vector<Entry> e;
    for (const auto &row : j.at("constraints")) {
        auto i = m.rl.size();
        m.row_names.push_back(row.value("name", "r" + std::to_string(i)));
        m.rl.push_back(bound(row.value("lb", json()), -inf));
        m.ru.push_back(bound(row.value("ub", json()), inf));
        for (auto it = row.at("coefficients").begin(); it != row.at("coefficients").end(); ++it) {
            if (!ids.count(it.key()))
                throw std::runtime_error("Unknown variable: " + it.key());
            e.push_back({int64_t(i), ids.at(it.key()), it.value().get<double>()});
        }
    }
    m.A = Sparse::build(m.rl.size(), m.names.size(), std::move(e));
    canonicalize(m);
    return m;
}
Model read_qplib(std::istream &in) {
    std::string line;
    std::istringstream current;
    auto token = [&]() -> std::string {
        std::string value;
        while (!(current >> value)) {
            if (!std::getline(in, line))
                throw std::runtime_error("Incomplete QPLIB input");
            line = line.substr(0, line.find('#'));
            current.clear();
            current.str(line);
        }
        return value;
    };
    auto integer = [&]() {
        auto text = token();
        size_t consumed = 0;
        auto value = std::stoll(text, &consumed);
        if (consumed != text.size())
            throw std::runtime_error("Invalid QPLIB integer token");
        return value;
    };
    auto extended_number = [&]() {
        auto text = token();
        size_t consumed = 0;
        auto value = std::stold(text, &consumed);
        if (consumed != text.size() || !std::isfinite(value))
            throw std::runtime_error("Invalid QPLIB numeric token");
        return value;
    };
    auto count = [&]() {
        auto value = integer();
        if (value < 0 || value > 100000000)
            throw std::runtime_error("Invalid QPLIB count");
        return value;
    };
    Model m;
    m.name = token();
    auto type = token();
    auto direction = token();
    if (type.size() != 3 || type[1] != 'C' || (type[2] != 'L' && type[2] != 'B') ||
        (type[0] != 'L' && type[0] != 'D' && type[0] != 'C' && type[0] != 'Q'))
        throw std::runtime_error("UNSUPPORTED QPLIB class: continuous linear constraints/box only");
    sense(m, direction);
    auto n = count();
    auto rows = type[2] == 'L' ? count() : 0;
    for (int64_t j = 0; j < n; ++j)
        addvar(m, "x" + std::to_string(j + 1));
    std::vector<Entry> qe, e;
    auto index = [&](int64_t size) {
        auto value = integer();
        if (value < 1 || value > size)
            throw std::runtime_error("Invalid QPLIB index");
        return value - 1;
    };
    if (type[0] != 'L') {
        auto terms = count();
        for (int64_t k = 0; k < terms; ++k) {
            auto i = index(n), j = index(n);
            auto v = number(token());
            if (i < j)
                throw std::runtime_error("QPLIB quadratic entries require lower triangle");
            if (i == j)
                m.q[i] += v;
            else {
                qe.push_back({i, j, v});
                qe.push_back({j, i, v});
            }
        }
    }
    auto cost = number(token());
    std::fill(m.c.begin(), m.c.end(), cost);
    auto overrides = count();
    for (int64_t k = 0; k < overrides; ++k) {
        auto j = index(n);
        m.c[j] = number(token());
    }
    m.offset = number(token());
    if (rows) {
        auto terms = count();
        for (int64_t k = 0; k < terms; ++k) {
            auto i = index(rows), j = index(n);
            e.push_back({i, j, number(token())});
        }
    }
    long double infinity_value = extended_number();
    if (!(infinity_value > 0))
        throw std::runtime_error("Invalid QPLIB infinity marker");
    auto bvalue = [&]() {
        auto value = extended_number();
        if (value >= infinity_value)
            return inf;
        if (value <= -infinity_value)
            return -inf;
        auto v = double(value);
        if (!std::isfinite(v))
            throw std::runtime_error("QPLIB finite bound overflow");
        return v;
    };
    auto bounds = [&](std::vector<double> &out, int64_t size) {
        auto def = bvalue();
        out.assign(size, def);
        auto count_override = count();
        for (int64_t k = 0; k < count_override; ++k) {
            auto j = index(size);
            out[j] = bvalue();
        }
    };
    if (rows) {
        bounds(m.rl, rows);
        bounds(m.ru, rows);
    }
    bounds(m.lb, n);
    bounds(m.ub, n);
    // Consume starts without treating an unverified supplied point as a solution.
    auto skip = [&](int64_t size) {
        (void)number(token());
        auto changes = count();
        for (int64_t k = 0; k < changes; ++k) {
            (void)index(size);
            (void)number(token());
        }
    };
    skip(n);
    if (rows)
        skip(rows);
    skip(n);
    auto changes = count();
    for (int64_t k = 0; k < changes; ++k) {
        auto j = index(n);
        m.names[j] = token();
    }
    for (int64_t i = 0; i < rows; ++i)
        m.row_names.push_back("r" + std::to_string(i + 1));
    changes = count();
    for (int64_t k = 0; k < changes; ++k) {
        auto i = index(rows);
        m.row_names[i] = token();
    }
    std::string trailing;
    if (current >> trailing)
        throw std::runtime_error("Unexpected QPLIB trailing data");
    while (std::getline(in, line))
        if (!trim(line.substr(0, line.find('#'))).empty())
            throw std::runtime_error("Unexpected QPLIB trailing data");
    m.A = Sparse::build(rows, n, std::move(e));
    m.Q = Sparse::build(n, n, std::move(qe));
    canonicalize(m);
    return m;
}
Model read_mps(std::istream &in) {
    Model m;
    std::string section, line, obj, rhs_set, range_set, bound_set, last_col;
    std::map<std::string, int64_t> vars, rows;
    std::vector<Entry> quadratic_entries;
    std::map<std::pair<int64_t, int64_t>, bool> quad_triangles;
    std::vector<char> rowtype;
    std::vector<double> rhs, ranges;
    std::vector<bool> has_range;
    std::vector<Entry> e;
    bool integer = false, ended = false;
    int lineno = 0;
    std::set<std::pair<std::string, std::string>> bound_seen;
    std::set<std::string> marker_variables, bounded_variables;
    auto var = [&](const std::string &s) {
        if (!vars.count(s))
            vars[s] = addvar(m, s);
        return vars.at(s);
    };
    auto pairs = [&](const std::vector<std::string> &t, size_t first, auto apply) {
        if (t.size() <= first || (t.size() - first) % 2)
            throw std::runtime_error("Expected row/value pairs");
        for (size_t k = first; k < t.size(); k += 2)
            apply(t[k], number(t[k + 1]));
    };
    while (std::getline(in, line)) {
        lineno++;
        if (trim(line).empty() || trim(line)[0] == '*')
            continue;
        auto t = words(line);
        // Fixed MPS permits embedded blanks inside its eight-character names.
        if (section == "ROWS" && line.size() >= 5 && line[0] == ' ' &&
            trim(line.substr(2, 2)).empty() &&
            (line.size() <= 12 || trim(line.substr(12)).empty())) {
            auto name = trim(line.substr(4, 8));
            if (!name.empty())
                t = {trim(line.substr(1, 1)), name};
        }
        if ((section == "COLUMNS" || section == "RHS" || section == "RANGES") &&
            line.size() >= 36 && line[0] == ' ' && trim(line.substr(12, 2)).empty() &&
            trim(line.substr(22, 2)).empty() && trim(line.substr(36, 3)).empty()) {
            auto first_value = trim(line.substr(24, 12));
            auto first_row = trim(line.substr(14, 8));
            bool numeric = false;
            try {
                (void)number(first_value);
                numeric = true;
            } catch (const std::exception &) {
            }
            if (numeric && !first_row.empty()) {
                t = {trim(line.substr(4, 8)), first_row, first_value};
                if (line.size() > 39) {
                    auto second_row = trim(line.substr(39, 8));
                    if (!second_row.empty()) {
                        t.push_back(second_row);
                        t.push_back(line.size() > 49 ? trim(line.substr(49, 12)) : "");
                    }
                }
                // Existing blank-name continuation handling expects an omitted token.
                if (t[0].empty())
                    t.erase(t.begin());
            }
        }
        if (t.empty())
            continue;
        try {
            std::string h = upper(t[0]);
            bool header = line[0] != ' ' && line[0] != '\t';
            if (header) {
                if (h == "NAME") {
                    if (t.size() > 1)
                        m.name = t[1];
                    section = h;
                    continue;
                }
                if (h == "ENDATA") {
                    ended = true;
                    break;
                }
                if (h == "OBJSENSE") {
                    section = h;
                    if (t.size() == 2)
                        sense(m, t[1]);
                    continue;
                }
                if (h == "ROWS" || h == "COLUMNS" || h == "RHS" || h == "RANGES" || h == "BOUNDS" ||
                    h == "QMATRIX" || h == "QUADOBJ") {
                    section = h;
                    continue;
                }
                throw std::runtime_error("UNSUPPORTED MPS section: " + h);
            }
            if (section == "OBJSENSE") {
                if (t.size() != 1)
                    throw std::runtime_error("Malformed OBJSENSE");
                sense(m, t[0]);
            } else if (section == "ROWS") {
                if (t.size() != 2 || t[0].size() != 1)
                    throw std::runtime_error("Malformed ROWS");
                char type = t[0][0];
                if (type == 'N') {
                    if (!obj.empty())
                        throw std::runtime_error("UNSUPPORTED multiple objective/free N rows");
                    obj = t[1];
                } else {
                    if (type != 'L' && type != 'G' && type != 'E')
                        throw std::runtime_error("Unknown row type");
                    if (rows.count(t[1]))
                        throw std::runtime_error("Duplicate row");
                    rows[t[1]] = rows.size();
                    m.row_names.push_back(t[1]);
                    rowtype.push_back(type);
                    rhs.push_back(0);
                    ranges.push_back(0);
                    has_range.push_back(false);
                }
            } else if (section == "COLUMNS") {
                if (t.size() == 3 && t[1] == "'MARKER'") {
                    if (t[2] == "'INTORG'" && !integer)
                        integer = true;
                    else if (t[2] == "'INTEND'" && integer)
                        integer = false;
                    else
                        throw std::runtime_error("Malformed integer marker");
                    continue;
                }
                size_t first = 1;
                std::string col = t[0];
                if (t.size() % 2 == 0) {
                    col = last_col;
                    first = 0;
                }
                if (col.empty())
                    throw std::runtime_error("Missing column name");
                last_col = col;
                auto j = var(col);
                if (integer) {
                    m.types[j] = VarType::Integer;
                    marker_variables.insert(col);
                }
                pairs(t, first, [&](const std::string &r, double v) {
                    if (r == obj)
                        m.c[j] += v;
                    else if (rows.count(r))
                        e.push_back({rows.at(r), j, v});
                    else
                        throw std::runtime_error("Unknown row: " + r);
                });
            } else if (section == "RHS" || section == "RANGES") {
                auto &selected = section == "RHS" ? rhs_set : range_set;
                size_t first = t.size() % 2 ? 1 : 0;
                if (first) {
                    if (selected.empty())
                        selected = t[0];
                    if (selected != t[0])
                        throw std::runtime_error("UNSUPPORTED multiple RHS/range sets");
                }
                pairs(t, first, [&](const std::string &r, double v) {
                    if (r == obj && section == "RHS") {
                        m.offset = -v;
                        return;
                    }
                    if (!rows.count(r))
                        throw std::runtime_error("Unknown RHS row");
                    auto i = rows.at(r);
                    if (section == "RHS")
                        rhs[i] = v;
                    else {
                        ranges[i] = v;
                        has_range[i] = true;
                    }
                });
            } else if (section == "BOUNDS") {
                // Fixed-format files may omit the bound-set field, including
                // the very first record. Preserve its column field explicitly.
                if (line.size() >= 22 && trim(line.substr(3, 1)).empty() &&
                    trim(line.substr(12, 2)).empty() && trim(line.substr(22, 2)).empty()) {
                    auto set = trim(line.substr(4, 8));
                    auto name = trim(line.substr(14, 8));
                    auto value = line.size() > 24 ? trim(line.substr(24, 12)) : "";
                    if (!name.empty()) {
                        if (set.empty())
                            set = bound_set.empty() ? "DEFAULT" : bound_set;
                        t = {trim(line.substr(1, 2)), set, name};
                        if (!value.empty())
                            t.push_back(value);
                    }
                }
                if (t.size() < 3 || t.size() > 4)
                    throw std::runtime_error("Malformed BOUNDS");
                if (bound_set.empty())
                    bound_set = t[1];
                if (bound_set != t[1])
                    throw std::runtime_error("UNSUPPORTED multiple bound sets");
                auto type = upper(t[0]);
                auto k = var(t[2]);
                bounded_variables.insert(t[2]);
                if (!bound_seen.insert({type, t[2]}).second)
                    throw std::runtime_error("Duplicate bound declaration");
                bool value =
                    type == "LO" || type == "UP" || type == "FX" || type == "LI" || type == "UI";
                if (t.size() != (value ? 4 : 3))
                    throw std::runtime_error("Wrong bound field count");
                double v = value ? number(t[3]) : 0;
                if (type == "LO" || type == "LI")
                    m.lb[k] = v;
                else if (type == "UP" || type == "UI") {
                    m.ub[k] = v;
                    if (v < 0 && !bound_seen.count({"LO", t[2]}) && !bound_seen.count({"LI", t[2]}))
                        m.lb[k] = -inf;
                } else if (type == "FX")
                    m.lb[k] = m.ub[k] = v;
                else if (type == "FR") {
                    m.lb[k] = -inf;
                    m.ub[k] = inf;
                } else if (type == "MI")
                    m.lb[k] = -inf;
                else if (type == "PL")
                    m.ub[k] = inf;
                else if (type == "BV") {
                    m.lb[k] = 0;
                    m.ub[k] = 1;
                    m.types[k] = VarType::Binary;
                } else
                    throw std::runtime_error("UNSUPPORTED bound type: " + type);
                if (type == "LI" || type == "UI")
                    m.types[k] = VarType::Integer;
            } else if (section == "QMATRIX" || section == "QUADOBJ") {
                if (t.size() != 3 || !vars.count(t[0]) || !vars.count(t[1]))
                    throw std::runtime_error("Invalid quadratic entry");
                auto i = vars.at(t[0]), j = vars.at(t[1]);
                if (i == j)
                    m.q[i] += number(t[2]);
                else {
                    if (section == "QUADOBJ") {
                        auto key = std::minmax(i, j);
                        bool triangle = i < j;
                        auto found = quad_triangles.find(key);
                        if (found != quad_triangles.end() && found->second != triangle)
                            throw std::runtime_error(
                                "Ambiguous QUADOBJ: provide one triangle only");
                        quad_triangles[key] = triangle;
                    }
                    quadratic_entries.push_back({i, j, number(t[2])});
                    if (section == "QUADOBJ")
                        quadratic_entries.push_back({j, i, number(t[2])});
                }
            } else
                throw std::runtime_error("Unexpected MPS data");
        } catch (const std::exception &ex) {
            throw std::runtime_error("MPS line " + std::to_string(lineno) + ": " + ex.what());
        }
    }
    m.Q = Sparse::build(m.c.size(), m.c.size(), std::move(quadratic_entries));
    if (!ended || integer || obj.empty())
        throw std::runtime_error("Incomplete MPS / unclosed integer marker");
    // Historical INTORG defaults to [0,1] when no BOUNDS record exists.
    // Any explicit bound record follows ordinary [0,+inf] defaults.
    for (const auto &variable : marker_variables)
        if (!bounded_variables.count(variable))
            m.ub[vars.at(variable)] = 1;
    for (size_t i = 0; i < rhs.size(); i++) {
        double l = -inf, u = inf;
        if (rowtype[i] == 'E')
            l = u = rhs[i];
        if (rowtype[i] == 'L')
            u = rhs[i];
        if (rowtype[i] == 'G')
            l = rhs[i];
        if (has_range[i]) {
            auto r = ranges[i];
            if (rowtype[i] == 'L')
                l = u - std::abs(r);
            else if (rowtype[i] == 'G')
                u = l + std::abs(r);
            else if (r >= 0)
                u = l + r;
            else
                l = u + r;
        }
        m.rl.push_back(l);
        m.ru.push_back(u);
    }
    m.A = Sparse::build(rows.size(), vars.size(), std::move(e));
    canonicalize(m);
    return m;
}
// Strict linear expression tokenizer; every input character must be consumed.
struct Expr {
    std::map<std::string, double> a;
    double constant = 0;
};
Expr expression(std::string s) {
    Expr e;
    static const std::regex token(
        R"(\s*([+-]|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-z_][A-Za-z_0-9.]*))");
    std::vector<std::string> t;
    std::smatch m;
    while (!trim(s).empty()) {
        if (!std::regex_search(s, m, token, std::regex_constants::match_continuous))
            throw std::runtime_error("Unsupported LP expression: " + s);
        t.push_back(m[1]);
        s = m.suffix();
    }
    size_t i = 0;
    bool first = true;
    while (i < t.size()) {
        double sign = 1;
        if (t[i] == "+" || t[i] == "-") {
            sign = t[i++] == "-" ? -1 : 1;
        } else if (!first)
            throw std::runtime_error("Missing sign in LP expression");
        if (i == t.size())
            throw std::runtime_error("Dangling sign");
        double a = 1;
        bool numeric = std::isdigit(t[i][0]) || t[i][0] == '.';
        if (numeric)
            a = number(t[i++]);
        if (i < t.size() && (std::isalpha(t[i][0]) || t[i][0] == '_'))
            e.a[t[i++]] += sign * a;
        else if (numeric)
            e.constant += sign * a;
        else
            throw std::runtime_error("Invalid linear term");
        first = false;
    }
    return e;
}
Model read_lp(std::istream &in) {
    Model m;
    std::map<std::string, int64_t> ids;
    std::vector<Entry> e;
    std::string line, section, objtext, pending;
    bool ended = false;
    auto var = [&](const std::string &s) {
        if (!ids.count(s))
            ids[s] = addvar(m, s);
        return ids.at(s);
    };
    auto striplabel = [](std::string s) {
        auto p = s.find(':');
        return p == s.npos ? s : s.substr(p + 1);
    };
    auto constraint = [&](std::string s) {
        static const std::regex cmp(R"((<=|>=|=))");
        std::smatch match;
        if (!std::regex_search(s, match, cmp))
            return false;
        auto name = "r" + std::to_string(m.rl.size());
        auto p = s.find(':');
        if (p != s.npos)
            name = trim(s.substr(0, p));
        s = striplabel(s);
        std::regex_search(s, match, cmp);
        auto lhs = expression(match.prefix()), rhs = expression(match.suffix());
        for (auto [k, v] : rhs.a)
            lhs.a[k] -= v;
        double b = rhs.constant - lhs.constant;
        auto i = m.rl.size();
        m.row_names.push_back(name);
        m.rl.push_back(match[1] == "<=" ? -inf : b);
        m.ru.push_back(match[1] == ">=" ? inf : b);
        for (auto [k, v] : lhs.a)
            e.push_back({int64_t(i), var(k), v});
        return true;
    };
    while (std::getline(in, line)) {
        auto comment = line.find('\\');
        if (comment != line.npos)
            line.resize(comment);
        line = trim(line);
        if (line.empty())
            continue;
        auto u = upper(line);
        if (u == "MINIMIZE" || u == "MINIMUM" || u == "MIN" || u == "MAXIMIZE" || u == "MAXIMUM" ||
            u == "MAX") {
            section = "obj";
            sense(m, u.substr(0, 3));
            continue;
        }
        if (u == "SUBJECT TO" || u == "SUCH THAT" || u == "ST" || u == "S.T.") {
            section = "rows";
            continue;
        }
        if (u == "BOUNDS" || u == "BINARY" || u == "BINARIES" || u == "GENERAL" ||
            u == "GENERALS" || u == "INTEGER" || u == "INTEGERS" || u == "END") {
            if (!pending.empty())
                throw std::runtime_error("Incomplete constraint");
            section = u;
            if (u == "END") {
                ended = true;
                break;
            }
            continue;
        }
        if (section == "obj")
            objtext += " " + striplabel(line);
        else if (section == "rows") {
            pending += " " + line;
            if (constraint(pending))
                pending.clear();
        } else if (section == "BOUNDS") {
            static const std::regex freepat(R"(^([A-Za-z_][A-Za-z_0-9.]*)\s+[Ff][Rr][Ee][Ee]$)");
            std::smatch a;
            if (std::regex_match(line, a, freepat)) {
                auto k = var(a[1]);
                m.lb[k] = -inf;
                m.ub[k] = inf;
                continue;
            }
            static const std::regex cmp(R"((<=|>=|=))");
            std::sregex_token_iterator it(line.begin(), line.end(), cmp, {-1, 0}), end;
            std::vector<std::string> t;
            for (; it != end; ++it)
                t.push_back(trim(*it));
            auto bval = [](std::string s) {
                auto u = upper(s);
                if (u == "INF" || u == "+INF" || u == "INFINITY" || u == "+INFINITY")
                    return inf;
                if (u == "-INF" || u == "-INFINITY")
                    return -inf;
                return number(s);
            };
            if (t.size() == 5 && t[1] == "<=" && t[3] == "<=") {
                auto k = var(t[2]);
                m.lb[k] = bval(t[0]);
                m.ub[k] = bval(t[4]);
            } else if (t.size() == 3) {
                bool leftvar = std::regex_match(t[0], std::regex(R"([A-Za-z_][A-Za-z_0-9.]*)")) &&
                               upper(t[0]) != "INF";
                auto k = var(leftvar ? t[0] : t[2]);
                auto v = bval(leftvar ? t[2] : t[0]);
                if (t[1] == "=")
                    m.lb[k] = m.ub[k] = v;
                else if ((t[1] == "<=" && leftvar) || (t[1] == ">=" && !leftvar))
                    m.ub[k] = v;
                else
                    m.lb[k] = v;
            } else
                throw std::runtime_error("Unsupported LP bound: " + line);
        } else if (section == "BINARY" || section == "BINARIES" || section == "GENERAL" ||
                   section == "GENERALS" || section == "INTEGER" || section == "INTEGERS") {
            for (auto s : words(line)) {
                auto k = var(s);
                if (section[0] == 'B') {
                    m.types[k] = VarType::Binary;
                    m.lb[k] = std::max(0., m.lb[k]);
                    m.ub[k] = std::min(1., m.ub[k]);
                } else
                    m.types[k] = VarType::Integer;
            }
        } else
            throw std::runtime_error("Unexpected LP line: " + line);
    }
    if (!ended || objtext.empty())
        throw std::runtime_error("Incomplete LP file");
    auto obj = expression(objtext);
    for (auto [k, v] : obj.a) {
        auto j = var(k);
        m.c[j] = v;
    }
    m.offset = obj.constant;
    m.A = Sparse::build(m.rl.size(), m.c.size(), std::move(e));
    canonicalize(m);
    return m;
}
json json_model(const Model &m) {
    json j = {{"name", m.name},
              {"sense", m.sense == 1 ? "min" : "max"},
              {"variables", json::array()},
              {"constraints", json::array()}};
    auto c = m.c, q = m.q;
    for (auto &v : c)
        v *= m.sense;
    for (auto &v : q)
        v *= m.sense;
    j["objective"] = {{"linear", c}, {"quadratic_diagonal", q}, {"offset", m.offset * m.sense}};
    if (!m.Q.value.empty()) {
        j["objective"]["quadratic_sparse"] = json::array();
        for (int64_t i = 0; i < m.Q.rows; ++i)
            for (auto k = m.Q.ptr[i]; k < m.Q.ptr[i + 1]; ++k)
                j["objective"]["quadratic_sparse"].push_back(
                    {json(i), json(m.Q.index[k]), json(m.Q.value[k] * m.sense)});
    }
    for (size_t k = 0; k < c.size(); k++)
        j["variables"].push_back({{"name", m.names[k]},
                                  {"lb", m.lb[k]},
                                  {"ub", m.ub[k]},
                                  {"type", m.types[k] == VarType::Continuous ? "continuous"
                                           : m.types[k] == VarType::Binary   ? "binary"
                                                                             : "integer"}});
    for (size_t i = 0; i < m.rl.size(); i++) {
        json a = json::object();
        for (auto k = m.A.ptr[i]; k < m.A.ptr[i + 1]; k++)
            a[m.names[m.A.index[k]]] = m.A.value[k];
        j["constraints"].push_back(
            {{"name", m.row_names[i]}, {"lb", m.rl[i]}, {"ub", m.ru[i]}, {"coefficients", a}});
    }
    return j;
}
} // namespace
Model read_model(const std::string &path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("Cannot open model: " + path);
    auto ext = upper(path.substr(path.find_last_of('.') + 1));
    if (ext == "QPLIB")
        return read_qplib(f);
    if (ext == "JSON")
        return read_json(f);
    if (ext == "MPS" || ext == "QPS")
        return read_mps(f);
    if (ext == "LP")
        return read_lp(f);
    throw std::runtime_error("Supported formats: .mps .qps .lp .json");
}
void write_model(const Model &m, const std::string &path) {
    m.validate();
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("Cannot write " + path);
    if (upper(path.substr(path.find_last_of('.') + 1)) == "JSON") {
        f << json_model(m).dump(2) << '\n';
        return;
    }
    if (upper(path.substr(path.find_last_of('.') + 1)) != "MPS")
        throw std::runtime_error("Convert output must be .json or .mps");
    for (const auto &name : m.names)
        if (name.find_first_of(" \t\r\n") != std::string::npos)
            throw std::runtime_error(
                "MPS export with whitespace in names requires fixed-format output; use JSON");
    for (const auto &name : m.row_names)
        if (name.find_first_of(" \t\r\n") != std::string::npos)
            throw std::runtime_error(
                "MPS export with whitespace in names requires fixed-format output; use JSON");
    f << std::setprecision(17) << "NAME          " << m.name << "\nOBJSENSE\n "
      << (m.sense == 1 ? "MIN" : "MAX") << "\nROWS\n N  OBJ\n";
    std::vector<double> rhs(m.rl.size());
    for (size_t i = 0; i < rhs.size(); i++) {
        char t = m.rl[i] == m.ru[i] ? 'E' : std::isfinite(m.ru[i]) ? 'L' : 'G';
        if (!std::isfinite(m.rl[i]) && !std::isfinite(m.ru[i]))
            throw std::runtime_error("Cannot serialize fully free row to supported MPS subset");
        rhs[i] = t == 'G' ? m.rl[i] : m.ru[i];
        f << ' ' << t << "  " << m.row_names[i] << '\n';
    }
    f << "COLUMNS\n";
    auto at = m.A.transpose();
    bool active = false;
    int mark = 0;
    for (size_t j = 0; j < m.c.size(); j++) {
        bool integral = m.types[j] != VarType::Continuous;
        if (integral != active) {
            f << " MARK" << mark++ << " 'MARKER' '" << (integral ? "INTORG" : "INTEND") << "'\n";
            active = integral;
        }
        f << " " << m.names[j] << " OBJ " << m.c[j] * m.sense << '\n';
        for (auto k = at.ptr[j]; k < at.ptr[j + 1]; k++)
            f << " " << m.names[j] << " " << m.row_names[at.index[k]] << " " << at.value[k] << '\n';
    }
    if (active)
        f << " MARK" << mark << " 'MARKER' 'INTEND'\n";
    f << "RHS\n RHS1 OBJ " << -m.offset * m.sense << '\n';
    for (size_t i = 0; i < rhs.size(); i++)
        f << " RHS1 " << m.row_names[i] << " " << rhs[i] << '\n';
    bool ranges = false;
    for (size_t i = 0; i < rhs.size(); i++)
        if (std::isfinite(m.rl[i]) && std::isfinite(m.ru[i]) && m.rl[i] != m.ru[i]) {
            if (!ranges) {
                f << "RANGES\n";
                ranges = true;
            }
            f << " RNG1 " << m.row_names[i] << " " << m.ru[i] - m.rl[i] << '\n';
        }
    f << "BOUNDS\n";
    for (size_t j = 0; j < m.c.size(); j++) {
        auto name = m.names[j];
        if (m.types[j] == VarType::Binary)
            f << " BV BND " << name << '\n';
        if (m.lb[j] == m.ub[j])
            f << " FX BND " << name << " " << m.lb[j] << '\n';
        else {
            if (std::isfinite(m.lb[j]))
                f << " LO BND " << name << " " << m.lb[j] << '\n';
            else
                f << " MI BND " << name << '\n';
            if (std::isfinite(m.ub[j]))
                f << " UP BND " << name << " " << m.ub[j] << '\n';
        }
    }
    if (m.is_qp()) {
        f << "QUADOBJ\n";
        for (size_t j = 0; j < m.q.size(); j++)
            if (m.q[j] != 0)
                f << " " << m.names[j] << " " << m.names[j] << " " << m.q[j] * m.sense << '\n';
    }
    if (!m.Q.value.empty())
        for (int64_t i = 0; i < m.Q.rows; ++i)
            for (auto k = m.Q.ptr[i]; k < m.Q.ptr[i + 1]; ++k)
                if (m.Q.index[k] >= i)
                    f << " " << m.names[i] << " " << m.names[m.Q.index[k]] << " "
                      << m.Q.value[k] * m.sense << '\n';
    f << "ENDATA\n";
}
std::string result_json(const Model &m, const Result &r) {
    auto a = r.accuracy;
    json j = {
        {"solver", "NIRYUKTI"},
        {"version", "0.2.0"},
        {"status", r.status},
        {"message", r.message},
        {"problem_type", m.is_mip()  ? (m.is_qp() ? "MIQP" : "MILP")
                         : m.is_qp() ? "QP"
                                     : "LP"},
        {"model",
         {{"name", m.name},
          {"variable_names", m.names},
          {"row_names", m.row_names},
          {"fingerprint", m.fingerprint()},
          {"rows", m.A.rows},
          {"columns", m.A.cols},
          {"nonzeros", m.A.value.size()}}},
        {"objective", a.finite ? json(m.sense * a.objective) : json()},
        {"accuracy",
         {{"finite", a.finite},
          {"primal_residual", a.primal},
          {"dual_residual", a.dual},
          {"relative_gap", a.gap},
          {"kkt_error", a.kkt},
          {"integrality", a.integrality},
          {"primal_absolute", a.primal_absolute},
          {"dual_absolute", a.dual_absolute},
          {"complementarity", a.complementarity},
          {"dual_bound", a.lower_bound}}},
        {"performance",
         {{"total_seconds", r.seconds},
          {"preprocess_seconds", r.preprocess_seconds},
          {"transfer_setup_seconds", r.transfer_seconds},
          {"iteration_seconds", r.iteration_seconds},
          {"verification_seconds", r.verification_seconds},
          {"iterations", r.iterations},
          {"weight_updates", r.weight_updates},
          {"polishing_iterations", r.polishing_iterations},
          {"polishing_attempts", r.polishing_attempts},
          {"operator_norm_estimate", r.operator_norm_estimate},
          {"restarts", r.restarts},
          {"rejected_steps", r.rejected_steps}}},
        {"hardware", {{"backend", r.backend}, {"device", r.device_name}, {"precision", "FP64"}}},
        {"presolve", {{"removed_rows", r.removed_rows}, {"removed_columns", r.removed_columns}}},
        {"primal", r.x},
        {"dual", r.y}};
    j["selection"] = {{"method", r.method_selected},
                      {"device", r.backend},
                      {"reason", r.device_reason},
                      {"estimated_gpu_bytes", r.estimated_gpu_bytes}};
    j["performance"]["monitor_checks"] = r.monitor_checks;
    j["performance"]["host_candidate_checks"] = r.host_candidate_checks;
    j["performance"]["skipped_candidate_checks"] = r.skipped_candidate_checks;
    j["gpu_execution"] = {{"index_bits", r.gpu_index_bits},
                          {"cuda_graphs", r.graph_execution},
                          {"matrix_precision", r.matrix_precision}};
    if (!r.infeasibility_ray.empty())
        j["certificate"] = {{"kind", "BOX_ROW_FARKAS"},
                            {"dual_ray", r.infeasibility_ray},
                            {"verified_margin", r.certificate_margin}};
    if (m.is_mip())
        j["mip"] = {
            {"nodes", r.nodes},
            {"strong_branch_probes", r.strong_branch_probes},
            {"cuts_added", r.cuts_added},
            {"pump_rounds", r.pump_rounds},
            {"rins_calls", r.rins_calls},
            {"heuristic_nodes", r.heuristic_nodes},
            {"bounds_tightened", r.bounds_tightened},
            {"nodes_remaining", r.nodes_remaining},
            {"best_bound", std::isfinite(r.best_bound) ? json(m.sense * r.best_bound) : json()},
            {"relative_gap", r.mip_gap}};
    return j.dump(2);
}
void write_solution(const Model &m, const Result &r, const std::string &path) {
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("Cannot write solution");
    f << result_json(m, r) << '\n';
}
Result read_solution(const Model &m, const std::string &path, bool allow_model_change) {
    std::ifstream f(path);
    json j;
    f >> j;
    if (j.at("model").at("fingerprint") != m.fingerprint()) {
        if (!allow_model_change)
            throw std::runtime_error("Solution/model fingerprint mismatch; parametric solves "
                                     "require --allow-model-change");
        if (j.at("model").at("variable_names") != m.names ||
            j.at("model").at("row_names") != m.row_names)
            throw std::runtime_error("Warm-start row/variable names or ordering mismatch");
    }
    Result r;
    r.x = j.at("primal").get<std::vector<double>>();
    r.y = j.at("dual").get<std::vector<double>>();
    r.status = j.at("status");
    if (j.contains("objective") && j["objective"].is_number())
        r.accuracy.objective = m.sense * j["objective"].get<double>();
    else if (j.contains("objective") && r.status == "OPTIMAL")
        throw std::runtime_error("Claimed optimal objective must be finite and numeric");
    if (j.contains("certificate")) {
        if (j.at("certificate").at("kind") != "BOX_ROW_FARKAS")
            throw std::runtime_error("Unsupported certificate kind");
        r.infeasibility_ray = j.at("certificate").at("dual_ray").get<std::vector<double>>();
    }
    return r;
}
} // namespace vantage
