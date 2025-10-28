import { useCallback, useState } from "react";
import { Upload, FileText, Loader2 } from "lucide-react";
import { Button } from "./ui/button";
import { cn } from "@/lib/utils";

interface ResumeUploadProps {
  onUpload: (file: File) => void;
  isProcessing: boolean;
}

export const ResumeUpload = ({ onUpload, isProcessing }: ResumeUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file && (file.type === "application/pdf" || file.name.endsWith(".pdf"))) {
      setSelectedFile(file);
    }
  }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  }, []);

  const handleUpload = () => {
    if (selectedFile) {
      onUpload(selectedFile);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto px-2 sm:px-4">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={cn(
          "relative border-2 border-dashed rounded-xl p-6 sm:p-8 md:p-12 transition-all duration-300",
          isDragging ? "border-primary bg-primary/5 scale-[1.02]" : "border-border bg-card",
          isProcessing && "pointer-events-none opacity-60"
        )}
      >
        {isProcessing ? (
          <div className="flex flex-col items-center gap-3 sm:gap-4">
            <Loader2 className="h-10 w-10 sm:h-12 sm:w-12 text-primary animate-spin" />
            <p className="text-base sm:text-lg font-medium px-2 text-center">Analyzing your resume with AI...</p>
            <p className="text-xs sm:text-sm text-muted-foreground">Finding the perfect job matches</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4 sm:gap-6">
            <div className="p-3 sm:p-4 rounded-full bg-primary/10">
              {selectedFile ? (
                <FileText className="h-10 w-10 sm:h-12 sm:w-12 text-primary" />
              ) : (
                <Upload className="h-10 w-10 sm:h-12 sm:w-12 text-primary" />
              )}
            </div>
            
            {selectedFile ? (
              <div className="text-center space-y-3 sm:space-y-4 w-full px-2">
                <p className="text-base sm:text-lg font-medium break-all">{selectedFile.name}</p>
                <p className="text-xs sm:text-sm text-muted-foreground">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
                <div className="flex flex-col sm:flex-row gap-3 justify-center">
                  <Button onClick={handleUpload} variant="hero" size="lg" className="w-full sm:w-auto">
                    Match Jobs with AI
                  </Button>
                  <Button 
                    onClick={() => setSelectedFile(null)} 
                    variant="outline"
                    size="lg"
                    className="w-full sm:w-auto"
                  >
                    Change File
                  </Button>
                </div>
              </div>
            ) : (
              <div className="text-center space-y-3 sm:space-y-4 w-full px-2">
                <h3 className="text-lg sm:text-xl font-semibold">Drop your resume here</h3>
                <p className="text-sm sm:text-base text-muted-foreground">or</p>
                <label htmlFor="resume-input" className="block">
                  <Button variant="hero" size="lg" asChild className="w-full sm:w-auto">
                    <span className="cursor-pointer">Browse Files</span>
                  </Button>
                  <input
                    id="resume-input"
                    type="file"
                    accept=".pdf"
                    onChange={handleFileInput}
                    className="hidden"
                  />
                </label>
                <p className="text-xs sm:text-sm text-muted-foreground">Supports PDF files up to 10MB</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
