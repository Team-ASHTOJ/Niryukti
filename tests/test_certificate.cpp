#include "json.hpp"
#include "sha256.hpp"
#include "vantage/vantage.hpp"
#include <iostream>
#include <stdexcept>
using namespace vantage;
using J = nlohmann::json;
void require(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
J report(const Model &m, J c) {
    c.erase("payload_sha256");
    c["payload_sha256"] = sha256(c.dump());
    return J::parse(verify_certificate_json(m, c.dump()));
}
Model model() {
    Model m;
    m.A = Sparse::build(1, 1, {{0, 0, 1}});
    m.c = {1};
    m.q = {0};
    m.lb = {0};
    m.ub = {10};
    m.rl = {2};
    m.ru = {inf};
    m.types = {VarType::Continuous};
    m.names = {"x"};
    m.row_names = {"r"};
    return m;
}
int main() {
    try {
        require(sha256("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "SHA empty");
        require(sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                "SHA abc");
        auto m = model();
        Options o;
        o.device = "cpu";
        o.method = "auto";
        auto r = solve(m, o);
        auto c = J::parse(certificate_json(m, r, o));
        require(report(m, c)["valid"], "LP replay");
        auto bad = c;
        bad["objective"] = 9;
        require(!report(m, bad)["valid"].get<bool>(),
                "objective independently checked after resealing");
        bad = c;
        bad["primal"]["x"] = 0;
        require(!report(m, bad)["valid"].get<bool>(), "primal independently checked");
        bad = c;
        bad["dual"]["r"] = 100;
        require(!report(m, bad)["valid"].get<bool>(), "dual independently checked");
        bad = c;
        bad["primal"]["x"] = 0;
        bad["objective"] = 0;
        require(J::parse(verify_certificate_json(m, bad.dump()))["reason"] == "primal feasibility",
                "precise primal tamper diagnostic");
        bad = c;
        bad["model_fingerprint"] = std::string(64, '0');
        require(!report(m, bad)["valid"].get<bool>(), "fingerprint tampering");
        bad = c;
        bad["verification"]["status"] = "FEASIBLE";
        require(!report(m, bad)["valid"].get<bool>(), "status independently checked");
        bad = c;
        bad["tolerances"]["feasibility"] = 1;
        require(!report(m, bad)["valid"].get<bool>(), "policy cannot be weakened");
        for (double scale : {1e-12, 1e12}) {
            auto scaled = m;
            scaled.c = {scale};
            auto sr = solve(scaled, o);
            auto sc = J::parse(certificate_json(scaled, sr, o));
            require(report(scaled, sc)["valid"], "coefficient scale");
        }
        auto empty = m;
        empty.A = Sparse::build(0, 0, {});
        empty.c.clear();
        empty.q.clear();
        empty.lb.clear();
        empty.ub.clear();
        empty.rl.clear();
        empty.ru.clear();
        empty.types.clear();
        empty.names.clear();
        empty.row_names.clear();
        auto er = solve(empty, o);
        require(report(empty, J::parse(certificate_json(empty, er, o)))["valid"], "empty model");
        auto mi = m;
        mi.types = {VarType::Integer};
        mi.rl = {2.5};
        Result incumbent;
        incumbent.x = {3};
        incumbent.y = {-1};
        incumbent.accuracy = verify(mi, incumbent.x, incumbent.y);
        incumbent.status = "OPTIMAL";
        auto mc = J::parse(certificate_json(mi, incumbent, o));
        require(mc["verification"]["status"] == "FEASIBLE_WITH_UNRESOLVED_GAP",
                "solver status cannot establish tree optimality");
        require(report(mi, mc)["valid"], "unresolved incumbent valid");
        auto corrupt_integer = mc;
        corrupt_integer["primal"]["x"] = 3.25;
        require(!report(mi, corrupt_integer)["valid"].get<bool>(),
                "integrality independently replayed");
        auto infeasible = m;
        infeasible.A = Sparse::build(2, 1, {{0, 0, 1}, {1, 0, 1}});
        infeasible.rl = {4, -inf};
        infeasible.ru = {inf, 2};
        infeasible.row_names = {"lower", "upper"};
        auto ir = solve(infeasible, o);
        auto ic = J::parse(certificate_json(infeasible, ir, o));
        require(report(infeasible, ic)["valid"], "Farkas replay");
        require(ic["verification"]["status"] == "INFEASIBLE", "Farkas status");
        std::cout << "Certificate numerical and resealed tamper checks passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
