#ifndef NIRYUKTI_C_API_H
#define NIRYUKTI_C_API_H
#include "vantage/c_api.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef vantage_session niryukti_session;
unsigned niryukti_abi_version(void);
niryukti_session *niryukti_open(const char *model_path);
void niryukti_close(niryukti_session *session);
const char *niryukti_last_error(void);
const char *niryukti_request(niryukti_session *session, const char *request_json);
#ifdef __cplusplus
}
#endif
#endif
