import { spawn, ChildProcess } from "child_process";
import path from "path";

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
    console.log("[django] Starting Django backend on port 8000...");
    
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

function startVite() {
  const rootDir = path.resolve(import.meta.dirname, "..");
  
  console.log("[vite] Starting Vite frontend on port 5000...");
  
  const viteProcess = spawn("npx", ["vite", "--host", "0.0.0.0", "--port", "5000"], {
    cwd: rootDir,
    stdio: "inherit",
    env: { ...process.env },
  });
  
  viteProcess.on("error", (err) => {
    console.error(`[vite] Failed to start: ${err.message}`);
  });
  
  viteProcess.on("exit", (code) => {
    console.log(`[vite] Process exited with code ${code}`);
    if (djangoProcess) {
      djangoProcess.kill();
    }
    process.exit(code || 0);
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
startVite();
