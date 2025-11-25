import type { Express, Request, Response, NextFunction } from "express";
import { createServer, type Server } from "http";
import http from "http";

export async function registerRoutes(app: Express): Promise<Server> {
  app.use('/api', (req: Request, res: Response, next: NextFunction) => {
    const bodyString = req.body && Object.keys(req.body).length > 0 
      ? JSON.stringify(req.body) 
      : '';
    
    const headers: Record<string, string> = {};
    for (const [key, value] of Object.entries(req.headers)) {
      if (value && key.toLowerCase() !== 'host' && key.toLowerCase() !== 'content-length') {
        headers[key] = Array.isArray(value) ? value.join(', ') : value;
      }
    }
    headers['host'] = 'localhost:8000';
    
    if (bodyString) {
      headers['content-length'] = Buffer.byteLength(bodyString).toString();
      headers['content-type'] = 'application/json';
    }

    const options = {
      hostname: 'localhost',
      port: 8000,
      path: '/api' + req.url,
      method: req.method,
      headers,
    };

    const proxyReq = http.request(options, (proxyRes) => {
      res.writeHead(proxyRes.statusCode || 500, proxyRes.headers);
      proxyRes.pipe(res);
    });

    proxyReq.on('error', (err) => {
      console.error('Proxy error:', err.message);
      res.status(502).json({
        error: 'Backend service unavailable',
        message: 'Django backend is not responding'
      });
    });

    if (bodyString) {
      proxyReq.write(bodyString);
    }

    proxyReq.end();
  });

  const httpServer = createServer(app);

  return httpServer;
}
