"""
Production Server Entry Point

Run with: python run.py
"""
import uvicorn
import argparse


def main():
    parser = argparse.ArgumentParser(description="AI Voice Detection API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           AI Voice Detection API Server                      ║
╠══════════════════════════════════════════════════════════════╣
║  🚀 Starting server...                                       ║
║  📍 URL: http://{args.host}:{args.port}                              ║
║  📚 Docs: http://{args.host}:{args.port}/docs                        ║
║  🔍 API: http://{args.host}:{args.port}/api/v1/detect                ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )


if __name__ == "__main__":
    main()
