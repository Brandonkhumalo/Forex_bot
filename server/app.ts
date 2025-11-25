import express, { type Express } from "express";
import { spawn, ChildProcess } from "child_process";
import path from "path";
import { createServer, type Server } from "http";
import { registerRoutes } from "./routes";

let djangoProcess: ChildProcess | null = null;

async function runMigrations(): Promise<void> {
  const backendDir = path.resolve(import.meta.dirname, "..", "backend");
  
  return new Promise((resolve) => {
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
      } else {
        console.log("[django] Migration failed with code", code);
      }
      resolve();
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
    
    djangoProcess.on("exit", (code) => {
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

process.on("SIGINT", () => {
  if (djangoProcess) {
    djangoProcess.kill();
  }
  process.exit();
});

process.on("SIGTERM", () => {
  if (djangoProcess) {
    djangoProcess.kill();
  }
  process.exit();
});

startDjango();

export default async function runApp(
  setup: (app: Express, server: Server) => Promise<void>,
) {
  const app = express();
  app.use(express.json());
  app.use(express.urlencoded({ extended: false }));

  const server = await registerRoutes(app);
  await setup(app, server);

  const port = 5000;
  server.listen(port, "0.0.0.0", () => {
    console.log(`${new Date().toLocaleTimeString()} [express] serving on port ${port}`);
  });

  return server;
}
