import { apiUrl } from "~/utils/endpoints";

interface SystemStatus {
  status: string;
  models_loaded: boolean;
  timestamp: string;
  upload_folder: string;
  threshold: number;
}

export async function checkStatus(): Promise<boolean> {
  try {
    const response = await fetch(`${apiUrl}/api/health`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      console.error("Server health check failed:", response.statusText);
      return false;
    }

    const data: SystemStatus = await response.json();

    if (data.status !== "ok" || data.models_loaded !== true) {
      console.error("Server health check failed:", data);
      return false;
    }

    console.log("API server connected successfully:", data);
    return true;
  } catch (error) {
    console.error("Server health check failed with exception:", error);
    return false;
  }
}

export async function getSystemInfo() {
  try {
    const response = await fetch(`${apiUrl}/api/system/status`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to get system info: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error getting system info:", error);
    throw error;
  }
}

export async function getModelsInfo() {
  try {
    const response = await fetch(`${apiUrl}/api/models/info`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to get models info: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error getting models info:", error);
    throw error;
  }
}

export async function analyzeWithThreshold(image: File, threshold?: number) {
  try {
    const formData = new FormData();
    formData.append("image", image);

    if (threshold !== undefined) {
      formData.append("threshold", threshold.toString());
    }

    const response = await fetch(`${apiUrl}/api/analyze/threshold`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(
        `Failed to analyze with threshold: ${response.statusText}`
      );
    }

    return await response.json();
  } catch (error) {
    console.error("Error analyzing with threshold:", error);
    throw error;
  }
}

export async function compareImages(image1: File, image2: File) {
  try {
    const formData = new FormData();
    formData.append("image1", image1);
    formData.append("image2", image2);

    const response = await fetch(`${apiUrl}/api/compare`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Failed to compare images: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error comparing images:", error);
    throw error;
  }
}
