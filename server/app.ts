import { type Server } from "node:http";
import { spawn, type ChildProcess } from "node:child_process";
import path from "node:path";

import express, {
  type Express,
  type Request,
  Response,
  NextFunction,
} from "express";

import { registerRoutes } from "./routes";

let djangoProcess: ChildProcess | null = null;

async function runMigrations(): Promise<void> {
  const backendDir = path.resolve(import.meta.dirname, "..", "backend");
  
  return new Promise((resolve, reject) => {
    console.log("[django] Running database migrations...");
    const migrate = spawn("python", ["manage.py", "migrate", "--run-syncdb"], {
      cwd: backendDir,
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env },
    });
    
    migrate.stdout?.on("data", (data) => {
      const output = data.toString().trim();
      if (output) console.log(`[django] ${output}`);
    });
    
    migrate.stderr?.on("data", (data) => {
      const output = data.toString().trim();
      if (output) console.log(`[django] ${output}`);
    });
    
    migrate.on("close", (code) => {
      if (code === 0) {
        console.log("[django] Migrations completed successfully");
        resolve();
      } else {
        console.log("[django] Migration failed with code", code);
        resolve();
      }
    });
    
    migrate.on("error", (err) => {
      console.error("[django] Migration error:", err.message);
      resolve();
    });
  });
}

function startDjango() {
  const backendDir = path.resolve(import.meta.dirname, "..", "backend");
  
  runMigrations().then(() => {
    console.log("[django] Starting Django backend server...");
    
    djangoProcess = spawn("python", ["manage.py", "runserver", "0.0.0.0:8000"], {
      cwd: backendDir,
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env },
    });
  
  djangoProcess.stdout?.on("data", (data) => {
    const output = data.toString().trim();
    if (output && !output.includes("Watching for file changes")) {
      console.log(`[django] ${output}`);
    }
  });
  
  djangoProcess.stderr?.on("data", (data) => {
    const output = data.toString().trim();
    if (output && !output.includes("Watching for file changes")) {
      console.log(`[django] ${output}`);
    }
  });
  
  djangoProcess.on("error", (err) => {
    console.error(`[django] Failed to start: ${err.message}`);
  });
  
    djangoProcess.on("exit", (code, signal) => {
      if (code !== null && code !== 0) {
        console.log(`[django] Process exited with code ${code}`);
        setTimeout(startDjango, 3000);
      }
    });
  });
}

process.on("exit", () => {
  if (djangoProcess) {
    djangoProcess.kill();
  }
});

export function log(message: string, source = "express") {
  const formattedTime = new Date().toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });

  console.log(`${formattedTime} [${source}] ${message}`);
}

export const app = express();

declare module 'http' {
  interface IncomingMessage {
    rawBody: unknown
  }
}
app.use(express.json({
  verify: (req, _res, buf) => {
    req.rawBody = buf;
  }
}));
app.use(express.urlencoded({ extended: false }));

app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined = undefined;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
      if (capturedJsonResponse) {
        logLine += ` :: ${JSON.stringify(capturedJsonResponse)}`;
      }

      if (logLine.length > 80) {
        logLine = logLine.slice(0, 79) + "…";
      }

      log(logLine);
    }
  });

  next();
});

export default async function runApp(
  setup: (app: Express, server: Server) => Promise<void>,
) {
  const server = await registerRoutes(app);

  app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";

    res.status(status).json({ message });
    throw err;
  });

  // importantly run the final setup after setting up all the other routes so
  // the catch-all route doesn't interfere with the other routes
  await setup(app, server);

  // Start Django backend
  startDjango();

  // ALWAYS serve the app on the port specified in the environment variable PORT
  // Other ports are firewalled. Default to 5000 if not specified.
  // this serves both the API and the client.
  // It is the only port that is not firewalled.
  const port = parseInt(process.env.PORT || '5000', 10);
  server.listen({
    port,
    host: "0.0.0.0",
    reusePort: true,
  }, () => {
    log(`serving on port ${port}`);
  });
}
