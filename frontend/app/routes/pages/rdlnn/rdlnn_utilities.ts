export async function checkStatus(): Promise<boolean> {
  const response = await fetch("http://localhost:5000/api/health", {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (response.ok) {
    const data = await response.json();
    if (data.status !== "ok" && data.models_loaded !== true) {
      console.error("Server health check failed:", data);
      return false;
    }
  } else {
    console.error("Server health check failed:", response.statusText);
    return false;
  }

  return true;
}
