import { useState, useEffect, use } from "react";
import { checkStatus } from "./rdlnn_utilities";
import ImageUpload from "~/components/utilities/imageUploadComponent";
import ImageDisplay from "~/components/utilities/imageDisplayComponent";
import { apiUrl } from "~/utils/endpoints";

interface Result {
  filename: String;
  original_image_base64: String;
  probability: Number;
  processing_time: Number;
  result: String;
  result_text: String;
  run_id: String;
  threshold: Number;
  timestamp: String;
}

export default function RDLNN() {
  const [data, setData] = useState(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [result, setResult] = useState<Result | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      if (await checkStatus()) {
        setModelLoaded(true);
      }
    };
    fetchData();
  }, []);

  useEffect(() => {
    if (selectedImage === null) {
      setResult(null);
    }
  }, [selectedImage]);

  const processImage = async (image: File) => {
    const formData = new FormData();
    formData.append("image", image);

    try {
      const response = await fetch(`${apiUrl}/api/detect`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const resultJSON = await response.json();
      setResult(resultJSON);
    } catch (error) {
      console.error("Error processing image:", error);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith("image/")) {
      setSelectedImage(file);
    }

    if (file) {
      processImage(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith("image/")) {
        setSelectedImage(file);
      }
    }
  };

  const removeImage = () => {
    setSelectedImage(null);
  };

  console.log(result);

  return (
    <div className="flex flex-col min-h-screen items-center justify-center bg-gunmetal text-silversand">
      {modelLoaded ? (
        <section className="flex flex-col items-center justify-center m-5 bg-gunmetal text-silversand">
          <div className="w-full max-w-md mx-auto">
            {!selectedImage ? (
              <ImageUpload
                isDragging={isDragging}
                handleDragLeave={handleDragLeave}
                handleDragOver={handleDragOver}
                handleDrop={handleDrop}
                handleFileUpload={handleFileUpload}
              />
            ) : (
              <ImageDisplay
                selectedImage={selectedImage}
                removeImage={removeImage}
              />
            )}
          </div>
        </section>
      ) : (
        <section>
          <p className="mt-4 text-lg">Loading...</p>
        </section>
      )}
      {result && (
        <div className="mt-8 p-6 bg-darkgunmetal rounded-lg shadow-lg w-full max-w-md">
          <h2 className="text-2xl font-bold mb-4 text-center text-aqua">
            Detection Results
          </h2>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="font-medium">Result:</span>
              <span
                className={
                  result.result === "defect"
                    ? "text-red-500 font-bold"
                    : "text-green-500 font-bold"
                }
              >
                {result.result.toUpperCase()}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="font-medium">Confidence:</span>
              <span>{(Number(result.probability) * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="font-medium">Processing Time:</span>
              <span>{Number(result.processing_time).toFixed(2)}ms</span>
            </div>
            <div className="flex justify-between">
              <span className="font-medium">Threshold:</span>
              <span>{Number(result.threshold).toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="font-medium">Run ID:</span>
              <span className="text-sm">
                {String(result.run_id).substring(0, 8)}...
              </span>
            </div>
            <div className="mt-4 pt-4 border-t border-gray-700">
              <div className="w-full bg-gray-700 rounded-full h-4">
                <div
                  className={`h-4 rounded-full ${
                    result.result === "defect" ? "bg-red-500" : "bg-green-500"
                  }`}
                  style={{ width: `${Number(result.probability) * 100}%` }}
                ></div>
              </div>
            </div>
          </div>
          <div className="mt-6 flex justify-center space-x-4">
            <button
              className="px-4 py-2 bg-aqua text-gunmetal bg-silversand cursor-pointer font-medium rounded-md hover:bg-teal-400 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-aqua shadow-md"
              onClick={() => {}}
            >
              Localize
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
