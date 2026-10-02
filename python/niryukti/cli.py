"""Installed CLI: native optimization, local HTTP service and HTML reports."""
import argparse
import json
import sys
from vantage.cli import main as native_main

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("serve", "report", "translate"):
        return native_main()
    if sys.argv[1] == "translate":
        parser=argparse.ArgumentParser(prog="niryukti translate",
                                       description="Offline controlled-English to JSON model")
        parser.add_argument("spec");parser.add_argument("--output")
        args=parser.parse_args(sys.argv[2:])
        from vantage.language import translate_file, TranslationError
        try:
            _, report=translate_file(args.spec,args.output)
        except TranslationError as error:
            print(json.dumps(dict(status="ERROR",errors=[dict(line=n,message=m) for n,m in error.errors]),indent=2))
            return 1
        print(json.dumps(dict(status="TRANSLATED",output=args.output,**report),indent=2));return 0
    if sys.argv[1] == "report":
        parser=argparse.ArgumentParser(prog="niryukti report")
        parser.add_argument("result");parser.add_argument("--output",required=True)
        parser.add_argument("--title",default="NIRYUKTI solve report")
        args=parser.parse_args(sys.argv[2:])
        from .report import write_report
        write_report(args.result,args.output,title=args.title)
        print(args.output);return 0
    parser=argparse.ArgumentParser(prog="niryukti serve")
    parser.add_argument("--host",default="127.0.0.1");parser.add_argument("--port",type=int,default=8090)
    parser.add_argument("--workers",type=int,default=2);parser.add_argument("--max-time",type=float,default=300)
    args=parser.parse_args(sys.argv[2:])
    from .service import make_server
    server=make_server(args.host,args.port,workers=args.workers,max_time=args.max_time)
    print(f"NIRYUKTI API http://{args.host}:{server.server_port}/v1 · authentication required",flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
    return 0
