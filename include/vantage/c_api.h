#ifndef VANTAGE_C_API_H
#define VANTAGE_C_API_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct vantage_session vantage_session;
/* ABI 1: caller owns handle; returned strings are borrowed until next call.
   Calls on one handle must be serialized. No C++ exception crosses the ABI. */
unsigned vantage_abi_version(void);
vantage_session *vantage_open(const char *model_path);
void vantage_close(vantage_session *session);
const char *vantage_last_error(void);
/* JSON request: action=solve or update, using the CLI session protocol. */
const char *vantage_request(vantage_session *session, const char *request_json);
#ifdef __cplusplus
}
#endif
#endif
