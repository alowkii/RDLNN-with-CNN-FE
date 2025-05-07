import { useState, useEffect } from "react";
import { checkStatus } from "./rdlnn_utilities";
import ImageUpload from "~/components/utilities/imageUploadComponent";
import ImageDisplay from "~/components/utilities/imageDisplayComponent";
import { apiUrl } from "~/utils/endpoints";

interface DetectionResult {
  filename: string;
  original_image_base64: string;
  probability: number;
  processing_time: number;
  result: string;
  result_text: string;
  run_id: string;
  threshold: number;
  timestamp: string;
}

interface LocalizationResult extends DetectionResult {
  forgery_map_base64?: string;
  regions?: any[];
}

export default function RDLNN() {
  const [modelLoaded, setModelLoaded] = useState(false);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [localizationResult, setLocalizationResult] =
    useState<LocalizationResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [showLocalization, setShowLocalization] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        if (await checkStatus()) {
          setModelLoaded(true);
        }
      } catch (error) {
        setErrorMessage(
          "Failed to connect to API server. Please check if the server is running."
        );
        console.error("API connection error:", error);
      }
    };
    fetchData();
  }, []);

  useEffect(() => {
    if (selectedImage === null) {
      setResult(null);
      setLocalizationResult(null);
      setShowLocalization(false);
    }
  }, [selectedImage]);

  const processImage = async (image: File, endpoint: string) => {
    setIsProcessing(true);
    setErrorMessage(null);

    const formData = new FormData();
    formData.append("image", image);

    try {
      const response = await fetch(`${apiUrl}${endpoint}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(
          `Server returned ${response.status}: ${response.statusText}`
        );
      }

      const resultJSON = await response.json();

      if (endpoint === "/api/rdlnn/detect") {
        setResult(resultJSON);
        setLocalizationResult(null);
      } else {
        setLocalizationResult(resultJSON);
        setShowLocalization(true);
      }
    } catch (error) {
      console.error("Error processing image:", error);
      setErrorMessage(
        `Error processing image: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith("image/")) {
      setSelectedImage(file);
      processImage(file, "/api/rdlnn/detect");
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
        processImage(file, "/api/rdlnn/detect");
      }
    }
  };

  const removeImage = () => {
    setSelectedImage(null);
  };

  const handleLocalize = () => {
    if (selectedImage) {
      processImage(selectedImage, "/api/localize");
    }
  };

  const handleSwitchView = () => {
    setShowLocalization(!showLocalization);
  };

  return (
    <div className="flex flex-col min-h-screen bg-gunmetal text-silversand">
      {errorMessage && (
        <div className="bg-red-500 text-white p-4 mb-4 rounded-md">
          {errorMessage}
        </div>
      )}

      {modelLoaded ? (
        <div className="container mx-auto px-4 py-8">
          <h1 className="text-3xl font-bold mb-6 text-center">
            RDLNN Forgery Detection
          </h1>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="flex flex-col">
              <h2 className="text-xl font-semibold mb-4">
                Upload an image for analysis
              </h2>
              {!selectedImage ? (
                <ImageUpload
                  isDragging={isDragging}
                  handleDragLeave={handleDragLeave}
                  handleDragOver={handleDragOver}
                  handleDrop={handleDrop}
                  handleFileUpload={handleFileUpload}
                />
              ) : (
                <div className="space-y-4">
                  <ImageDisplay
                    selectedImage={selectedImage}
                    removeImage={removeImage}
                  />

                  {isProcessing && (
                    <div className="flex justify-center items-center h-16">
                      <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-silversand"></div>
                    </div>
                  )}

                  {(result || localizationResult) && !isProcessing && (
                    <div className="flex flex-col gap-4">
                      <button
                        className="px-4 py-2 bg-silversand text-gunmetal font-medium rounded-md hover:bg-coolgrey transition-colors duration-200"
                        onClick={handleSwitchView}
                        disabled={!localizationResult}
                      >
                        {showLocalization
                          ? "Show Detection Results"
                          : "Show Localization Results"}
                      </button>

                      {!localizationResult && (
                        <button
                          className="px-4 py-2 bg-silversand text-gunmetal font-medium rounded-md hover:bg-coolgrey transition-colors duration-200"
                          onClick={handleLocalize}
                          disabled={
                            isProcessing ||
                            !result ||
                            result.result.toLowerCase() !== "forged"
                          }
                        >
                          Localize Forgery
                        </button>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="flex flex-col">
              {result && !isProcessing && !showLocalization && (
                <div className="p-6 bg-black bg-opacity-25 rounded-lg shadow-lg">
                  <h2 className="text-2xl font-bold mb-4 text-center">
                    Detection Results
                  </h2>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="font-medium">Result:</span>
                      <span
                        className={
                          result.result.toLowerCase() === "forged"
                            ? "text-red-500 font-bold"
                            : "text-green-500 font-bold"
                        }
                      >
                        {result.result.toUpperCase()}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-medium">Confidence:</span>
                      <span>
                        {(Number(result.probability) * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-medium">Threshold:</span>
                      <span>{Number(result.threshold).toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-medium">Processing Time:</span>
                      <span>{Number(result.processing_time).toFixed(2)}s</span>
                    </div>
                    <div className="mt-4 pt-4 border-t border-gray-700">
                      <div className="w-full bg-gray-700 rounded-full h-4">
                        <div
                          className={`h-4 rounded-full ${
                            result.result.toLowerCase() === "forged"
                              ? "bg-red-500"
                              : "bg-green-500"
                          }`}
                          style={{
                            width: `${Number(result.probability) * 100}%`,
                          }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {localizationResult && showLocalization && (
                <div className="p-6 bg-black bg-opacity-25 rounded-lg shadow-lg">
                  <h2 className="text-2xl font-bold mb-4 text-center">
                    Forgery Localization
                  </h2>

                  {localizationResult.forgery_map_base64 ? (
                    <div className="space-y-4">
                      <p className="text-center mb-2">
                        The highlighted areas indicate potential forgery regions
                      </p>
                      <div className="flex justify-center">
                        <img
                          src={`data:image/png;base64,${localizationResult.forgery_map_base64}`}
                          alt="Forgery Map"
                          className="max-h-72 object-contain"
                        />
                      </div>

                      {localizationResult.regions &&
                        localizationResult.regions.length > 0 && (
                          <div className="mt-4">
                            <h3 className="text-lg font-semibold mb-2">
                              Detected Regions:
                            </h3>
                            <div className="overflow-y-auto max-h-40">
                              <ul className="space-y-1">
                                {localizationResult.regions.map(
                                  (region, index) => (
                                    <li key={index}>
                                      Region {index + 1}: x={region.x}, y=
                                      {region.y}, width={region.width}, height=
                                      {region.height}
                                    </li>
                                  )
                                )}
                              </ul>
                            </div>
                          </div>
                        )}
                    </div>
                  ) : (
                    <p className="text-center text-yellow-400">
                      No forgery map available. The image might be authentic or
                      the forgery couldn't be localized.
                    </p>
                  )}
                </div>
              )}

              {!result &&
                !localizationResult &&
                !isProcessing &&
                selectedImage && (
                  <div className="flex items-center justify-center h-full">
                    <p className="text-lg text-silversand opacity-50">
                      Processing image... Results will appear here.
                    </p>
                  </div>
                )}

              {!selectedImage && (
                <div className="flex items-center justify-center h-full">
                  <p className="text-lg text-silversand opacity-50">
                    Upload an image to see detection results.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center h-screen">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-silversand mb-4"></div>
          <p className="text-xl">Connecting to API server...</p>
          {errorMessage && (
            <p className="text-red-400 mt-4 max-w-md text-center">
              {errorMessage}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
