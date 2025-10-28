import { useState } from "react";
import { Sparkles } from "lucide-react";
import { ResumeUpload } from "@/components/ResumeUpload";
import { JobMatchCard } from "@/components/JobMatchCard";
import heroBg from "@/assets/hero-bg.jpg";

const Index = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [showMatches, setShowMatches] = useState(false);

  // Mock job data - would come from backend in production
  const mockJobs = [
    {
      title: "Senior Frontend Developer",
      company: "TechCorp Inc",
      location: "San Francisco, CA",
      salary: "$120k - $160k",
      type: "Full-time",
      matchScore: 95,
      description: "We're looking for an experienced frontend developer to join our growing team. You'll work on cutting-edge React applications...",
      skills: ["React", "TypeScript", "Tailwind CSS", "Node.js", "Git"]
    },
    {
      title: "Full Stack Engineer",
      company: "StartupXYZ",
      location: "Remote",
      salary: "$100k - $140k",
      type: "Full-time",
      matchScore: 88,
      description: "Join our mission to revolutionize the industry. We need someone who can work across the entire stack...",
      skills: ["React", "Python", "PostgreSQL", "AWS", "Docker"]
    },
    {
      title: "React Developer",
      company: "Digital Agency Co",
      location: "New York, NY",
      salary: "$90k - $120k",
      type: "Contract",
      matchScore: 82,
      description: "Looking for a talented React developer to build beautiful, responsive web applications for our clients...",
      skills: ["React", "JavaScript", "CSS", "Figma", "REST APIs"]
    }
  ];

  const handleUpload = (file: File) => {
    console.log("Uploaded file:", file.name);
    setIsProcessing(true);
    
    // Simulate AI processing
    setTimeout(() => {
      setIsProcessing(false);
      setShowMatches(true);
    }, 3000);
  };

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section 
        className="relative py-12 sm:py-16 md:py-20 px-4 sm:px-6 overflow-hidden"
        style={{ background: 'var(--gradient-hero)' }}
      >
        <div 
          className="absolute inset-0 opacity-30"
          style={{ backgroundImage: `url(${heroBg})`, backgroundSize: 'cover', backgroundPosition: 'center' }}
        />
        
        <div className="relative max-w-7xl mx-auto text-center space-y-6 sm:space-y-8">
          <div className="inline-flex items-center gap-2 px-3 sm:px-4 py-1.5 sm:py-2 rounded-full bg-primary/10 border border-primary/20">
            <Sparkles className="h-3 w-3 sm:h-4 sm:w-4 text-primary" />
            <span className="text-xs sm:text-sm font-medium">AI-Powered Job Matching</span>
          </div>
          
          <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight px-2">
            Find Your Perfect
            <span className="block bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              Career Match
            </span>
          </h1>
          
          <p className="text-base sm:text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto px-4">
            Upload your resume and let our advanced NLP technology match you with opportunities that align perfectly with your skills and experience.
          </p>
        </div>
      </section>

      {/* Upload Section */}
      <section className="py-12 sm:py-16 px-4 sm:px-6">
        <div className="max-w-7xl mx-auto">
          {!showMatches ? (
            <div className="space-y-6 sm:space-y-8">
              <div className="text-center space-y-2">
                <h2 className="text-2xl sm:text-3xl font-bold px-2">Upload Your Resume</h2>
                <p className="text-sm sm:text-base text-muted-foreground px-4">
                  Our AI will analyze your skills and experience in seconds
                </p>
              </div>
              <ResumeUpload onUpload={handleUpload} isProcessing={isProcessing} />
            </div>
          ) : (
            <div className="space-y-6 sm:space-y-8">
              <div className="text-center space-y-2">
                <h2 className="text-2xl sm:text-3xl font-bold px-2">Your Top Matches</h2>
                <p className="text-sm sm:text-base text-muted-foreground px-4">
                  We found {mockJobs.length} jobs that match your profile
                </p>
              </div>
              
              <div className="grid gap-4 sm:gap-6 max-w-4xl mx-auto">
                {mockJobs.map((job, idx) => (
                  <JobMatchCard key={idx} {...job} />
                ))}
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Features Section */}
      <section className="py-12 sm:py-16 md:py-20 px-4 sm:px-6 bg-secondary/30">
        <div className="max-w-7xl mx-auto">
          <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-6 sm:gap-8 text-center">
            <div className="space-y-3 px-2">
              <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mx-auto">
                <Sparkles className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-lg sm:text-xl font-semibold">AI-Powered Matching</h3>
              <p className="text-sm sm:text-base text-muted-foreground">
                Advanced NLP analyzes your resume to find the perfect opportunities
              </p>
            </div>
            <div className="space-y-3 px-2">
              <div className="w-12 h-12 rounded-full bg-accent/10 flex items-center justify-center mx-auto">
                <Sparkles className="h-6 w-6 text-accent" />
              </div>
              <h3 className="text-lg sm:text-xl font-semibold">Real-Time Results</h3>
              <p className="text-sm sm:text-base text-muted-foreground">
                Get matched with jobs in seconds, not days
              </p>
            </div>
            <div className="space-y-3 px-2 sm:col-span-2 md:col-span-1">
              <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mx-auto">
                <Sparkles className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-lg sm:text-xl font-semibold">Smart Recommendations</h3>
              <p className="text-sm sm:text-base text-muted-foreground">
                Only see jobs that truly match your skills and goals
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Index;
