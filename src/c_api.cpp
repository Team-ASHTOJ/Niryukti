#include "vantage/c_api.h"
#include "session.hpp"
#include <mutex>
#include <cstdio>
struct vantage_session {
    vantage::Session engine;
    std::string output;
    explicit vantage_session(const char *path) : engine(path) {}
};
namespace {
thread_local char last_error[2048] = {};
// The existing solver has process-global interrupt state; serialize native calls.
std::mutex api_mutex;
}
extern "C" {
unsigned vantage_abi_version(void) { return 1; }
vantage_session *vantage_open(const char *path) {
    try {
        std::lock_guard<std::mutex> lock(api_mutex);
        last_error[0]='\0';
        if (!path) throw std::runtime_error("Null model path");
        return new vantage_session(path);
    } catch(const std::exception &e) { std::snprintf(last_error,sizeof(last_error),"%s",e.what()); return nullptr; }
    catch(...) { std::snprintf(last_error,sizeof(last_error),"Native session creation failed"); return nullptr; }
}
void vantage_close(vantage_session *session) { delete session; }
const char *vantage_last_error(void) { return last_error; }
const char *vantage_request(vantage_session *session,const char *request) {
    try {
        std::lock_guard<std::mutex> lock(api_mutex);
        last_error[0]='\0';
        if (!session || !request) throw std::runtime_error("Null session or request");
        session->output=session->engine.request(nlohmann::json::parse(request)).dump();
        return session->output.c_str();
    } catch(const std::exception &e) { std::snprintf(last_error,sizeof(last_error),"%s",e.what()); return nullptr; }
    catch(...) { std::snprintf(last_error,sizeof(last_error),"Native request failed"); return nullptr; }
}
}
