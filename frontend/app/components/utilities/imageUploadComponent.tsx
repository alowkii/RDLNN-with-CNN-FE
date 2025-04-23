import React from "react";

interface ImageUploadProps {
  isDragging: boolean;
  handleDragLeave: (e: React.DragEvent) => void;
  handleDragOver: (e: React.DragEvent) => void;
  handleDrop: (e: React.DragEvent) => void;
  handleFileUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

export default function ImageUpload({
  isDragging,
  handleDragLeave,
  handleDragOver,
  handleDrop,
  handleFileUpload,
}: ImageUploadProps) {
  return (
    <div
      className={`border-2 border-dashed rounded-lg p-8 transition-colors ${
        isDragging
          ? "border-blue-500 bg-opacity-20 bg-blue-900"
          : "border-silversand"
      }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="flex flex-col items-center justify-center space-y-4">
        {/* Simple Upload Icon */}
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
          htmlFor="imageInput"
          className="bg-gunmetal text-silversand border-2 border-silversand rounded-lg p-2 cursor-pointer hover:bg-silversand hover:text-gunmetal transition-colors"
        >
          Browse files
        </label>
        <input
          type="file"
          id="imageInput"
          accept="image/*"
          onChange={handleFileUpload}
          className="hidden"
        />
        <p className="text-xs text-silversand opacity-70">
          Supported formats: JPG, PNG
        </p>
      </div>
    </div>
  );
}
