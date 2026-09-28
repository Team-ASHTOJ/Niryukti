#include "vantage/vantage.hpp"
#include <iostream>
#include <stdexcept>
using namespace vantage;
void check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
int main() {
    Model m; m.c={1,1};m.q={0,0};m.lb={0,0};m.ub={10,10};
    m.types={VarType::Continuous,VarType::Continuous};m.names={"x","y"};m.row_names={"demand"};
    m.rl={4};m.ru={inf};m.A=Sparse::build(1,2,{{0,0,1},{0,1,2}});
    Options o;o.method="auto";
    auto cpu=advise_model(m,o,{});
    check(cpu.method=="simplex" && cpu.device=="cpu","compact LP decision");
    o.initial_basis={0};check(advise_model(m,o,{}).method=="dual-simplex","basis-aware decision");
    o.initial_basis.clear();o.checkpoint_path="state.json";
    check(advise_model(m,o,{}).method=="pdhg","checkpoint-compatible decision");
    o.checkpoint_path.clear();
    o.power_iterations=10;check(advise_model(m,o,{}).method=="pdhg","explicit first-order controls");
    o.power_iterations=0;
    auto sparse=m;sparse.A.value.resize(100000,1);
    Hardware gpu{true,8e9,8e9};auto large=advise_model(sparse,o,gpu);
    check(large.device=="cuda" && large.method=="pdhg","large workload GPU decision");
    check(advise_model(sparse,o,{true,1000,8e9}).device=="cpu","memory guard");
    o.device="cpu";check(advise_model(sparse,o,gpu).device=="cpu","explicit CPU respected");
    o.device="auto";o.method="rhpdhg";
    check(advise_model(m,o,{}).method=="rhpdhg","explicit method respected");
    o.method="auto";auto solved=solve(m,o);
    check(solved.status=="OPTIMAL" && solved.accuracy.kkt<=o.tol,"automatic solution verified");
    check(solved.device_reason.find("Advisor:")!=std::string::npos,"decision exposed");
    std::cout<<"Advisor policy and verified automatic solve: PASS\n";
}
