#pragma once
#include "../src/session.hpp"
#include <iostream>
inline int run_session(const std::string &path) {
    using nlohmann::json;
    vantage::Session session(path);
    std::cout << json({{"status","READY"},{"fingerprint",session.fingerprint()}}).dump() << std::endl;
    std::string line;
    while (std::getline(std::cin,line)) {
        try {
            auto request=json::parse(line);
            if (request.value("action", "") == "close") return 0;
            std::cout << session.request(request).dump() << std::endl;
        } catch(const std::exception &e) {
            std::cout << json({{"status","ERROR"},{"message",e.what()}}).dump() << std::endl;
        }
    }
    return 0;
}
