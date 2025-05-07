import { useState } from "react";
import { compareImages } from "./rdlnn_utilities";
import { apiUrl } from "~/utils/endpoints";

interface ComparisonResult {
  image1: {
    filename: string;
    result: string;
    probability: number;
  };
  image2: {
    filename: string;
    result: string;
    probability: number;
  };
  similarity?: number;
  processing_time: number;
  timestamp: number;
  image1_base64: string;
  image2_base64: string;
}

export default function ImageComparison() {
  const [image1, setImage1] = useState<File | null>(null);
  const [image2, setImage2] = useState<File | null>(null);
  const [isDragging1, setIsDragging1] = useState(false);
  const [isDragging2, setIsDragging2] = useState(false);
  const [result, setResult] = useState<ComparisonResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleFile1Upload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith("image/")) {
      setImage1(file);
    }
  };

  const handleFile2Upload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith("image/")) {
      setImage2(file);
    }
  };

  const handleDragOver1 = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging1(true);
  };

  const handleDragOver2 = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging2(true);
  };

  const handleDragLeave1 = () => {
    setIsDragging1(false);
  };

  const handleDragLeave2 = () => {
    setIsDragging2(false);
  };

  const handleDrop1 = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging1(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith("image/")) {
        setImage1(file);
      }
    }
  };

  const handleDrop2 = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging2(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith("image/")) {
        setImage2(file);
      }
    }
  };

  const removeImage1 = () => {
    setImage1(null);
  };

  const removeImage2 = () => {
    setImage2(null);
  };

  const handleCompare = async () => {
    if (!image1 || !image2) {
      setErrorMessage("Please select two images to compare");
      return;
    }

    setIsProcessing(true);
    setErrorMessage(null);

    try {
      const formData = new FormData();
      formData.append("image1", image1);
      formData.append("image2", image2);

      const response = await fetch(`${apiUrl}/api/compare`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(
          `Server returned ${response.status}: ${response.statusText}`
        );
      }

      const resultJSON = await response.json();
      setResult(resultJSON);
    } catch (error) {
      console.error("Error comparing images:", error);
      setErrorMessage(
        `Error comparing images: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    } finally {
      setIsProcessing(false);
    }
  };

  const getSimilarityText = (similarity: number | undefined) => {
    if (similarity === undefined) return "Unable to calculate similarity";

    if (similarity > 0.9) return "Very High Similarity";
    if (similarity > 0.7) return "High Similarity";
    if (similarity > 0.5) return "Moderate Similarity";
    if (similarity > 0.3) return "Low Similarity";
    return "Very Low Similarity";
  };

  const getSimilarityColor = (similarity: number | undefined) => {
    if (similarity === undefined) return "text-yellow-400";

    if (similarity > 0.9) return "text-red-500";
    if (similarity > 0.7) return "text-orange-500";
    if (similarity > 0.5) return "text-yellow-400";
    if (similarity > 0.3) return "text-green-400";
    return "text-green-500";
  };

  return (
    <div className="container mx-auto px-4 py-8 bg-gunmetal text-silversand">
      <h1 className="text-3xl font-bold mb-6 text-center">Image Comparison</h1>

      {errorMessage && (
        <div className="bg-red-500 text-white p-4 mb-4 rounded-md">
          {errorMessage}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        <div className="flex flex-col">
          <h2 className="text-xl font-semibold mb-4">Image 1</h2>
          {!image1 ? (
            <div
              className={`border-2 border-dashed rounded-lg p-8 transition-colors ${
                isDragging1
                  ? "border-blue-500 bg-opacity-20 bg-blue-900"
                  : "border-silversand"
              }`}
              onDragOver={handleDragOver1}
              onDragLeave={handleDragLeave1}
              onDrop={handleDrop1}
            >
              <div className="flex flex-col items-center justify-center space-y-4">
                <div className="w-12 h-12 flex items-center justify-center">
                  <svg
                    className="w-10 h-10 text-silversand"
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                </div>
                <div className="text-center">
                  <p className="text-lg font-medium text-silversand">
                    Drag and drop an image
                  </p>
                  <p className="text-sm text-silversand opacity-70">or</p>
                </div>
                <label
                  htmlFor="imageInput1"
                  className="bg-gunmetal text-silversand border-2 border-silversand rounded-lg p-2 cursor-pointer hover:bg-silversand hover:text-gunmetal transition-colors"
                >
                  Browse files
                </label>
                <input
                  type="file"
                  id="imageInput1"
                  accept="image/*"
                  onChange={handleFile1Upload}
                  className="hidden"
                />
                <p className="text-xs text-silversand opacity-70">
                  Supported formats: JPG, PNG
                </p>
              </div>
            </div>
          ) : (
            <div className="relative border border-silversand rounded-lg p-4 bg-opacity-10 bg-silversand">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center">
                  <span className="text-sm text-gunmetal truncate max-w-xs">
                    {image1.name}
                  </span>
                </div>
                <button
                  onClick={removeImage1}
                  className="p-1 rounded-full bg-opacity-20 bg-gunmetal hover:bg-opacity-40 transition-colors cursor-pointer"
                  aria-label="Remove image"
                >
                  <svg
                    className="w-4 h-4 text-silversand"
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </button>
              </div>
              <div className="flex justify-center">
                <img
                  src={URL.createObjectURL(image1)}
                  alt="Selected 1"
                  className="rounded-md max-h-72 object-cover bg-black bg-opacity-20"
                />
              </div>
            </div>
          )}
        </div>

        <div className="flex flex-col">
          <h2 className="text-xl font-semibold mb-4">Image 2</h2>
          {!image2 ? (
            <div
              className={`border-2 border-dashed rounded-lg p-8 transition-colors ${
                isDragging2
                  ? "border-blue-500 bg-opacity-20 bg-blue-900"
                  : "border-silversand"
              }`}
              onDragOver={handleDragOver2}
              onDragLeave={handleDragLeave2}
              onDrop={handleDrop2}
            >
              <div className="flex flex-col items-center justify-center space-y-4">
                <div className="w-12 h-12 flex items-center justify-center">
                  <svg
                    className="w-10 h-10 text-silversand"
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                </div>
                <div className="text-center">
                  <p className="text-lg font-medium text-silversand">
                    Drag and drop an image
                  </p>
                  <p className="text-sm text-silversand opacity-70">or</p>
                </div>
                <label
                  htmlFor="imageInput2"
                  className="bg-gunmetal text-silversand border-2 border-silversand rounded-lg p-2 cursor-pointer hover:bg-silversand hover:text-gunmetal transition-colors"
                >
                  Browse files
                </label>
                <input
                  type="file"
                  id="imageInput2"
                  accept="image/*"
                  onChange={handleFile2Upload}
                  className="hidden"
                />
                <p className="text-xs text-silversand opacity-70">
                  Supported formats: JPG, PNG
                </p>
              </div>
            </div>
          ) : (
            <div className="relative border border-silversand rounded-lg p-4 bg-opacity-10 bg-silversand">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center">
                  <span className="text-sm text-gunmetal truncate max-w-xs">
                    {image2.name}
                  </span>
                </div>
                <button
                  onClick={removeImage2}
                  className="p-1 rounded-full bg-opacity-20 bg-gunmetal hover:bg-opacity-40 transition-colors cursor-pointer"
                  aria-label="Remove image"
                >
                  <svg
                    className="w-4 h-4 text-silversand"
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </button>
              </div>
              <div className="flex justify-center">
                <img
                  src={URL.createObjectURL(image2)}
                  alt="Selected 2"
                  className="rounded-md max-h-72 object-cover bg-black bg-opacity-20"
                />
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="flex justify-center mb-8">
        <button
          className="px-6 py-3 bg-silversand text-gunmetal font-bold rounded-md hover:bg-metallicsilver transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
          onClick={handleCompare}
          disabled={!image1 || !image2 || isProcessing}
        >
          {isProcessing ? (
            <span className="flex items-center">
              <svg
                className="animate-spin -ml-1 mr-3 h-5 w-5 text-gunmetal"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
              Comparing...
            </span>
          ) : (
            "Compare Images"
          )}
        </button>
      </div>

      {result && (
        <div className="p-6 bg-black bg-opacity-25 rounded-lg shadow-lg">
          <h2 className="text-2xl font-bold mb-6 text-center">
            Comparison Results
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-6">
            <div className="flex flex-col">
              <h3 className="text-lg font-semibold mb-4">
                Image 1: {result.image1.filename}
              </h3>
              <div className="flex justify-center mb-4">
                <img
                  src={`data:image/jpeg;base64,${result.image1_base64}`}
                  alt="Image 1"
                  className="rounded-md max-h-72 object-cover bg-black bg-opacity-20"
                />
              </div>
              <div className="mt-4 space-y-2">
                <div className="flex justify-between">
                  <span className="font-medium">Result:</span>
                  <span
                    className={
                      result.image1.result.toLowerCase() === "forged"
                        ? "text-red-500 font-bold"
                        : "text-green-500 font-bold"
                    }
                  >
                    {result.image1.result.toUpperCase()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">Confidence:</span>
                  <span>{(result.image1.probability * 100).toFixed(2)}%</span>
                </div>
              </div>
            </div>

            <div className="flex flex-col">
              <h3 className="text-lg font-semibold mb-4">
                Image 2: {result.image2.filename}
              </h3>
              <div className="flex justify-center mb-4">
                <img
                  src={`data:image/jpeg;base64,${result.image2_base64}`}
                  alt="Image 2"
                  className="rounded-md max-h-72 object-cover bg-black bg-opacity-20"
                />
              </div>
              <div className="mt-4 space-y-2">
                <div className="flex justify-between">
                  <span className="font-medium">Result:</span>
                  <span
                    className={
                      result.image2.result.toLowerCase() === "forged"
                        ? "text-red-500 font-bold"
                        : "text-green-500 font-bold"
                    }
                  >
                    {result.image2.result.toUpperCase()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">Confidence:</span>
                  <span>{(result.image2.probability * 100).toFixed(2)}%</span>
                </div>
              </div>
            </div>
          </div>

          <div className="border-t border-gray-700 pt-6">
            <h3 className="text-xl font-semibold mb-4 text-center">
              Similarity Analysis
            </h3>

            {result.similarity !== undefined ? (
              <div className="flex flex-col items-center space-y-4">
                <div className="text-2xl font-bold">
                  <span className={getSimilarityColor(result.similarity)}>
                    {getSimilarityText(result.similarity)}
                  </span>
                </div>

                <div className="w-full max-w-md">
                  <div className="text-center mb-2">
                    Similarity Score: {(result.similarity * 100).toFixed(2)}%
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-4">
                    <div
                      className="h-4 rounded-full bg-blue-500"
                      style={{ width: `${result.similarity * 100}%` }}
                    ></div>
                  </div>
                </div>

                <div className="text-center max-w-lg mt-4 text-sm opacity-80">
                  <p>
                    High similarity with different forgery results may indicate
                    that one image was derived from the other with
                    manipulations. Low similarity may indicate unrelated images
                    or extensive modifications.
                  </p>
                </div>
              </div>
            ) : (
              <div className="text-center text-yellow-400">
                Unable to calculate similarity between images. The images may be
                too different or in incompatible formats.
              </div>
            )}

            <div className="mt-6 text-center text-sm opacity-70">
              Processing Time: {result.processing_time.toFixed(2)}s
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
