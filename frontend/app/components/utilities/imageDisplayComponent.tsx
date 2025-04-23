import type { ReactElement } from "react";

interface ImageDisplayProps {
  selectedImage: File | null;
  removeImage: () => void;
}

export default function ImageDisplay({
  selectedImage,
  removeImage,
}: ImageDisplayProps) {
  return (
    <div className="relative border border-silversand rounded-lg p-4 bg-opacity-10 bg-silversand">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <span className="text-sm text-gunmetal truncate max-w-xs">
            {selectedImage?.name}
          </span>
        </div>
        <button
          onClick={removeImage}
          className="p-1 rounded-full bg-opacity-20 bg-gunmetal hover:bg-opacity-40 transition-colors cursor-pointer"
          aria-label="Remove image"
        >
          {/* Simple X Icon */}
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
        {selectedImage && (
          <img
            src={URL.createObjectURL(selectedImage)}
            alt="Selected"
            className="rounded-md max-h-72 object-cover bg-black bg-opacity-20"
          />
        )}
      </div>
    </div>
  );
}
