import { Link } from "react-router";

export default function Footer() {
  return (
    <footer className="bg-black bg-opacity-30 p-6 mt-16">
      <div className="container mx-auto">
        <div className="flex flex-col md:flex-row justify-between">
          <div className="mb-4 md:mb-0">
            <h3 className="text-lg font-semibold mb-2">About This Tool</h3>
            <p className="text-metallicsilver text-sm max-w-md">
              This image forgery detection system uses deep learning techniques
              to identify and localize manipulated regions in digital images.
              The RDLNN model is designed to detect various types of forgeries
              including copy-move, splicing, and more. The DWT and DyWT models
              are also available for comparison, providing a comprehensive
              analysis of image integrity. This tool is intended for research
              only.
            </p>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-2">API Information</h3>
            <p className="text-metallicsilver text-sm">
              The backend API provides comprehensive forgery detection
              capabilities. Visit the{" "}
              <Link to="/api" className="text-silversand hover:underline">
                API Documentation
              </Link>{" "}
              to learn more about the available endpoints.
            </p>
          </div>
        </div>
        <div className="border-t border-metallicsilver mt-6 pt-6 text-center text-sm text-metallicsilver">
          Image Forgery Detection System &copy; {new Date().getFullYear()}
        </div>
      </div>
    </footer>
  );
}
