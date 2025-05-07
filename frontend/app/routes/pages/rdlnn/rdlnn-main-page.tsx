import { useState } from "react";
import RDLNN from "./rdlnn-model";
import ImageComparison from "./image-comparision";

export default function RDLNNPage() {
  const [currentTab, setCurrentTab] = useState<"detection" | "comparison">(
    "detection"
  );

  return (
    <div className="min-h-screen bg-gunmetal text-silversand">
      {/* Page Header */}
      <div className="bg-black bg-opacity-30 p-6 mb-8">
        <div className="container mx-auto">
          <h1 className="text-4xl font-bold mb-2">Image Forgery Detection</h1>
          <p className="text-metallicsilver mb-6">
            Powered by RDLNN (Regression Deep Learning Neural Network)
          </p>

          {/* Tab Navigation */}
          <div className="flex border-b border-metallicsilver">
            <button
              className={`py-2 px-6 font-medium transition-colors duration-200 ${
                currentTab === "detection"
                  ? "border-b-2 border-silversand text-silversand"
                  : "text-metallicsilver hover:text-silversand"
              }`}
              onClick={() => setCurrentTab("detection")}
            >
              Forgery Detection
            </button>
            <button
              className={`py-2 px-6 font-medium transition-colors duration-200 ${
                currentTab === "comparison"
                  ? "border-b-2 border-silversand text-silversand"
                  : "text-metallicsilver hover:text-silversand"
              }`}
              onClick={() => setCurrentTab("comparison")}
            >
              Image Comparison
            </button>
          </div>
        </div>
      </div>

      {/* Tab Content */}
      <div className="container mx-auto px-4">
        {currentTab === "detection" ? <RDLNN /> : <ImageComparison />}
      </div>
    </div>
  );
}
