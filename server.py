#!/usr/bin/env python3
import http.server
import socketserver
import webbrowser
import sys
import os

def main():
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            port = 8000
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"Server running on http://localhost:{port}")
            webbrowser.open(f"http://localhost:{port}/index.html")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
    except OSError as e:
        if e.errno == 48:
            print(f"Port {port} is already in use! Try: python server.py {port + 1}")
        else:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()