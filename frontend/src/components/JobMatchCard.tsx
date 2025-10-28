import { Briefcase, MapPin, DollarSign, Clock } from "lucide-react";
import { Card } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";

interface JobMatchCardProps {
  title: string;
  company: string;
  location: string;
  salary: string;
  type: string;
  matchScore: number;
  description: string;
  skills: string[];
}

export const JobMatchCard = ({
  title,
  company,
  location,
  salary,
  type,
  matchScore,
  description,
  skills,
}: JobMatchCardProps) => {
  return (
    <Card className="p-4 sm:p-6 hover:shadow-[var(--shadow-card)] transition-all duration-300 group">
      <div className="flex flex-col sm:flex-row justify-between items-start mb-4 gap-3 sm:gap-0">
        <div className="flex-1 w-full sm:w-auto">
          <h3 className="text-lg sm:text-xl font-semibold mb-1 group-hover:text-primary transition-colors">
            {title}
          </h3>
          <p className="text-sm sm:text-base text-muted-foreground font-medium">{company}</p>
        </div>
        <div className="flex flex-col items-start sm:items-end gap-2 self-end sm:self-auto">
          <div className="flex items-center gap-2 sm:gap-2">
            <div className="text-left sm:text-right">
              <p className="text-xs text-muted-foreground">Match Score</p>
              <p className="text-xl sm:text-2xl font-bold text-accent">{matchScore}%</p>
            </div>
            <div 
              className="w-14 h-14 sm:w-16 sm:h-16 rounded-full flex items-center justify-center flex-shrink-0"
              style={{
                background: `conic-gradient(hsl(var(--accent)) ${matchScore * 3.6}deg, hsl(var(--muted)) 0deg)`
              }}
            >
              <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-card flex items-center justify-center">
                <span className="text-xs sm:text-sm font-bold">{matchScore}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="flex flex-col sm:flex-row sm:flex-wrap gap-2 sm:gap-3 mb-4 text-xs sm:text-sm text-muted-foreground">
        <div className="flex items-center gap-1">
          <MapPin className="h-3 w-3 sm:h-4 sm:w-4 flex-shrink-0" />
          <span className="truncate">{location}</span>
        </div>
        <div className="flex items-center gap-1">
          <DollarSign className="h-3 w-3 sm:h-4 sm:w-4 flex-shrink-0" />
          <span>{salary}</span>
        </div>
        <div className="flex items-center gap-1">
          <Clock className="h-3 w-3 sm:h-4 sm:w-4 flex-shrink-0" />
          <span>{type}</span>
        </div>
      </div>

      <p className="text-xs sm:text-sm text-foreground/80 mb-4 line-clamp-2">{description}</p>

      <div className="flex flex-wrap gap-2 mb-4">
        {skills.slice(0, 5).map((skill, idx) => (
          <Badge key={idx} variant="secondary" className="text-xs">
            {skill}
          </Badge>
        ))}
        {skills.length > 5 && (
          <Badge variant="outline" className="text-xs">+{skills.length - 5} more</Badge>
        )}
      </div>

      <div className="flex flex-col sm:flex-row gap-2 sm:gap-3">
        <Button variant="hero" className="flex-1 text-sm sm:text-base">
          <Briefcase className="h-3 w-3 sm:h-4 sm:w-4" />
          Apply Now
        </Button>
        <Button variant="outline" className="w-full sm:w-auto text-sm sm:text-base">View Details</Button>
      </div>
    </Card>
  );
};
