import { useState, useCallback, useEffect } from 'react';
import { motion } from 'motion/react';
import { Sparkles, Upload, TrendingUp, Brain, Zap, Github } from 'lucide-react';
import { Toaster, toast } from 'sonner';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface Candidate {
  id: number;
  name: string;
  file_hash: string;
  github_username?: string | null;
  score: number;
  semanticMatch: number;
  mlStrength: number;
  decision: string;
  audit_status?: {
    flagged: boolean;
    copy_paste_percentage: number;
    stuffed_keywords: string[];
    warnings: string[];
  };
}

export default function App() {
  const [jobDescription, setJobDescription] = useState('');
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isUpdatingJD, setIsUpdatingJD] = useState(false);

  // Chat Sidebar States
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<{ role: 'user' | 'assistant'; content: string }[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isSendingChat, setIsSendingChat] = useState(false);

  // Dynamic Weights States (What-If Weight Simulator)
  const [wSem, setWSem] = useState(0.60);
  const [wKw, setWKw] = useState(0.25);
  const [wExp, setWExp] = useState(0.15);
  const [wMatch, setWMatch] = useState(0.60);
  const [wStrength, setWStrength] = useState(0.40);

  // GitHub Integration States
  const [isGithubModalOpen, setIsGithubModalOpen] = useState(false);
  const [githubModalLoading, setGithubModalLoading] = useState(false);
  const [activeGithubCandidate, setActiveGithubCandidate] = useState<Candidate | null>(null);
  const [githubStats, setGithubStats] = useState<any | null>(null);
  const [manualGithubInput, setManualGithubInput] = useState('');
  const [isLinkingGithub, setIsLinkingGithub] = useState(false);

  // Blind Hiring Mode States
  const [isBlindMode, setIsBlindMode] = useState(true);
  const [revealedIds, setRevealedIds] = useState<number[]>([]);

  // Fetch real candidates from FastAPI with optional custom weights
  const fetchCandidates = useCallback(async (
    wSemVal = 0.60,
    wKwVal = 0.25,
    wExpVal = 0.15,
    wMatchVal = 0.60,
    wStrengthVal = 0.40
  ) => {
    try {
      const query = `?w_sem=${wSemVal}&w_kw=${wKwVal}&w_exp=${wExpVal}&w_match=${wMatchVal}&w_strength=${wStrengthVal}`;
      const res = await fetch(`${API_BASE_URL}/api/candidates${query}`);
      if (res.ok) {
        const data = await res.json();
        const mapped = data.candidates.map((c: any) => ({
          id: c.rank,
          name: c.name,
          file_hash: c.file_hash,
          github_username: c.github_username,
          score: Math.round(c.hire_probability * 100),
          semanticMatch: Math.round(c.semantic_score * 100),
          mlStrength: Math.round(c.candidate_strength * 100),
          decision: c.decision.replace(/[\[\]]/g, ''), // removes [ ] around decision
          audit_status: c.audit_status
        }));
        setCandidates(mapped);
      }
    } catch (err) {
      console.error('Failed to fetch candidates', err);
    }
  }, []);

  // Debounced recalculation when sliders change
  useEffect(() => {
    const handler = setTimeout(() => {
      fetchCandidates(wSem, wKw, wExp, wMatch, wStrength);
    }, 200);

    return () => {
      clearTimeout(handler);
    };
  }, [wSem, wKw, wExp, wMatch, wStrength, fetchCandidates]);

  const handleUpdateJD = async () => {
    if (!jobDescription) return;
    setIsUpdatingJD(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jd: jobDescription }),
      });
      if (res.ok) {
        toast.success('Job description updated successfully!');
        await fetchCandidates(wSem, wKw, wExp, wMatch, wStrength);
      } else {
        toast.error('Failed to update job description.');
      }
    } catch (err) {
      console.error(err);
      toast.error('Network error. Failed to update Job Description.');
    } finally {
      setIsUpdatingJD(false);
    }
  };

  const handleFileUpload = async (file: File) => {
    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${API_BASE_URL}/api/resumes/upload`, {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        const data = await res.json();
        if (data.message.includes('already uploaded')) {
          toast.warning(data.message);
        } else {
          toast.success(data.message);
        }
        await fetchCandidates(wSem, wKw, wExp, wMatch, wStrength);
      } else {
        const errData = await res.json();
        toast.error(errData.detail || 'Failed to upload resume.');
      }
    } catch (err) {
      console.error(err);
      toast.error('Network error. Failed to connect to server.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleBulkFileUploads = async (files: File[]) => {
    setIsUploading(true);
    const total = files.length;
    let successCount = 0;
    let skippedCount = 0;
    let failedCount = 0;

    const toastId = toast.loading(`Uploading 1 of ${total} resumes...`, {
      description: "Processing sequentially to prevent rate limits.",
    });

    for (let i = 0; i < total; i++) {
      const file = files[i];
      // Update loading status
      toast.loading(`Uploading ${i + 1} of ${total}: ${file.name}...`, { 
        id: toastId,
        description: "Processing sequentially to prevent rate limits.",
      });

      try {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch(`${API_BASE_URL}/api/resumes/upload`, {
          method: 'POST',
          body: formData,
        });

        if (res.ok) {
          const data = await res.json();
          if (data.message.includes('already uploaded')) {
            skippedCount++;
          } else {
            successCount++;
          }
        } else {
          failedCount++;
        }
      } catch (err) {
        console.error(err);
        failedCount++;
      }

      // Add a small 1-second delay between requests to guarantee rate limits are not triggered
      if (i < total - 1) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    }

    // Refresh candidate leaderboard
    await fetchCandidates(wSem, wKw, wExp, wMatch, wStrength);
    setIsUploading(false);

    // Show consolidated results
    const summaryMsg = `Bulk upload finished. ${successCount} added, ${skippedCount} skipped, ${failedCount} failed.`;
    if (failedCount > 0) {
      toast.warning(summaryMsg, { id: toastId, duration: 6000 });
    } else {
      toast.success(summaryMsg, { id: toastId, duration: 6000 });
    }
  };

  const handleSendChatMessage = async () => {
    if (!chatInput.trim()) return;
    const userQuery = chatInput;
    setChatInput('');
    
    // Add user message to state
    const updatedMessages = [...chatMessages, { role: 'user' as const, content: userQuery }];
    setChatMessages(updatedMessages);
    setIsSendingChat(true);
    
    try {
      const res = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userQuery,
          history: chatMessages
        }),
      });
      
      if (res.ok) {
        const data = await res.json();
        setChatMessages([...updatedMessages, { role: 'assistant' as const, content: data.reply }]);
      } else {
        toast.error('Failed to get response from AI Recruiter Assistant.');
      }
    } catch (err) {
      console.error(err);
      toast.error('Network error. Failed to connect to chat agent.');
    } finally {
      setIsSendingChat(false);
    }
  };

  const handleSemChange = (val: number) => {
    const newSem = val / 100;
    const remaining = 1.0 - newSem;
    const currentSum = wKw + wExp;
    if (currentSum > 0) {
      setWSem(newSem);
      setWKw(remaining * (wKw / currentSum));
      setWExp(remaining * (wExp / currentSum));
    } else {
      setWSem(newSem);
      setWKw(remaining / 2);
      setWExp(remaining / 2);
    }
  };

  const handleKwChange = (val: number) => {
    const newKw = val / 100;
    const remaining = 1.0 - newKw;
    const currentSum = wSem + wExp;
    if (currentSum > 0) {
      setWKw(newKw);
      setWSem(remaining * (wSem / currentSum));
      setWExp(remaining * (wExp / currentSum));
    } else {
      setWKw(newKw);
      setWSem(remaining / 2);
      setWExp(remaining / 2);
    }
  };

  const handleExpChange = (val: number) => {
    const newExp = val / 100;
    const remaining = 1.0 - newExp;
    const currentSum = wSem + wKw;
    if (currentSum > 0) {
      setWExp(newExp);
      setWSem(remaining * (wSem / currentSum));
      setWKw(remaining * (wKw / currentSum));
    } else {
      setWExp(newExp);
      setWSem(remaining / 2);
      setWKw(remaining / 2);
    }
  };

  const resetWeights = () => {
    setWSem(0.60);
    setWKw(0.25);
    setWExp(0.15);
    setWMatch(0.60);
    setWStrength(0.40);
    toast.success('Weights reset to AI defaults.');
  };

  const handleOpenGithubModal = async (candidate: Candidate) => {
    setActiveGithubCandidate(candidate);
    setIsGithubModalOpen(true);
    setManualGithubInput(candidate.github_username || '');
    
    if (!candidate.github_username) {
      setGithubStats(null);
      return;
    }

    setGithubModalLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/candidates/${candidate.file_hash}/github`);
      if (res.ok) {
        const data = await res.json();
        setGithubStats(data.stats);
      } else {
        toast.error('Failed to load GitHub stats.');
        setGithubStats(null);
      }
    } catch (err) {
      console.error(err);
      toast.error('Failed to connect to GitHub API.');
      setGithubStats(null);
    } finally {
      setGithubModalLoading(false);
    }
  };

  const handleLinkGithubSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!activeGithubCandidate) return;

    setIsLinkingGithub(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/candidates/${activeGithubCandidate.file_hash}/github`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ github_username: manualGithubInput }),
      });

      if (res.ok) {
        toast.success('GitHub profile updated!');
        await fetchCandidates(wSem, wKw, wExp, wMatch, wStrength);
        
        const updatedCandidate = { ...activeGithubCandidate, github_username: manualGithubInput || null };
        setActiveGithubCandidate(updatedCandidate);
        
        if (manualGithubInput.trim()) {
          setGithubModalLoading(true);
          const statsRes = await fetch(`${API_BASE_URL}/api/candidates/${activeGithubCandidate.file_hash}/github`);
          if (statsRes.ok) {
            const statsData = await statsRes.json();
            setGithubStats(statsData.stats);
          } else {
            setGithubStats(null);
          }
          setGithubModalLoading(false);
        } else {
          setGithubStats(null);
        }
      } else {
        toast.error('Failed to update GitHub handle.');
      }
    } catch (err) {
      console.error(err);
      toast.error('Network error. Failed to save handle.');
    } finally {
      setIsLinkingGithub(false);
    }
  };

  const getDecisionStyle = (decision: string) => {
    switch (decision) {
      case 'INTERVIEW':
        return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/50 shadow-[0_0_15px_rgba(16,185,129,0.3)]';
      case 'PHONE SCREEN':
        return 'bg-amber-500/20 text-amber-400 border-amber-500/50 shadow-[0_0_15px_rgba(245,158,11,0.3)]';
      case 'AUTO-REJECT':
        return 'bg-red-500/10 text-red-400/60 border-red-500/30';
      default:
        return 'bg-slate-500/10 text-slate-400 border-slate-500/30';
    }
  };

  return (
    <div className="dark min-h-screen w-full bg-gradient-to-br from-slate-950 via-purple-950/20 to-slate-950 text-foreground">
      <Toaster richColors position="top-right" />
      <div className="mx-auto max-w-[1800px] px-4 py-6 md:px-8 md:py-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 flex items-center justify-between"
        >
          <div className="flex items-center gap-3">
            <div className="relative">
              <Sparkles className="h-8 w-8 text-purple-500" />
              <div className="absolute inset-0 animate-pulse blur-xl">
                <Sparkles className="h-8 w-8 text-purple-500" />
              </div>
            </div>
            <h1 className="bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent text-3xl font-bold">
              AI Recruiter
            </h1>
          </div>
          <div className="flex items-center gap-6">
            {/* Blind Review Toggle */}
            <div className="flex items-center gap-3 rounded-xl border border-white/5 bg-slate-900/40 px-4 py-2 text-xs font-semibold backdrop-blur-sm shadow-md">
              <span className="text-slate-300">Blind Review Mode</span>
              <button
                onClick={() => {
                  setIsBlindMode(!isBlindMode);
                  toast.info(!isBlindMode ? "Blind Review active. Anonymizing candidate names." : "Blind Review deactivated. Showing all names.");
                }}
                className={`relative inline-flex h-5 w-10 items-center rounded-full transition-colors outline-none focus:ring-1 focus:ring-purple-500/50 ${
                  isBlindMode ? 'bg-purple-600' : 'bg-slate-700'
                }`}
              >
                <span
                  className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                    isBlindMode ? 'translate-x-5.5' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>

            <motion.button
              onClick={() => setIsChatOpen(true)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="flex items-center gap-2 rounded-xl border border-purple-500/60 bg-purple-900/90 px-4 py-2 text-sm font-bold text-purple-100 backdrop-blur-sm transition-all hover:bg-purple-800 hover:shadow-[0_0_20px_rgba(168,85,247,0.3)] hover:text-white"
            >
              <Brain className="h-4 w-4" />
              Recruiter Assistant
            </motion.button>
          </div>
        </motion.div>

        {/* Main Layout */}
        <div className="grid gap-6 lg:grid-cols-[35%_1fr]">
          {/* LEFT PANEL - Input & Controls */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="space-y-6"
          >
            {/* Job Description Section */}
            <div className="group rounded-2xl border border-white/10 bg-slate-900/40 p-6 shadow-2xl backdrop-blur-xl transition-all hover:border-purple-500/30 hover:shadow-[0_0_30px_rgba(168,85,247,0.15)]">
              <h3 className="mb-4 flex items-center gap-2 text-purple-300 font-semibold">
                <Brain className="h-5 w-5" />
                Job Description
              </h3>
              <textarea
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
                placeholder="Paste or type the target job requirements here..."
                className="h-48 w-full resize-none rounded-xl border border-white/5 bg-slate-950/60 px-4 py-3 text-slate-200 placeholder-slate-500 outline-none ring-purple-500/50 backdrop-blur-sm transition-all focus:border-purple-500/50 focus:ring-2"
              />
              <motion.button
                onClick={handleUpdateJD}
                disabled={isUpdatingJD}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="mt-4 w-full rounded-xl border border-purple-500/50 bg-gradient-to-r from-purple-600/90 to-cyan-600/90 px-6 py-3 transition-all hover:border-purple-400 hover:shadow-[0_0_25px_rgba(168,85,247,0.5)] active:shadow-[0_0_15px_rgba(168,85,247,0.3)] disabled:opacity-50"
              >
                <span className="flex items-center justify-center gap-2 font-medium text-white">
                  <Zap className="h-4 w-4" />
                  {isUpdatingJD ? 'Updating...' : 'Update JD'}
                </span>
              </motion.button>
            </div>

            {/* Resume Upload Section */}
            <div className="group rounded-2xl border border-white/10 bg-slate-900/40 p-6 shadow-2xl backdrop-blur-xl transition-all hover:border-cyan-500/30 hover:shadow-[0_0_30px_rgba(6,182,212,0.15)]">
              <h3 className="mb-4 flex items-center gap-2 text-cyan-300 font-semibold">
                <Upload className="h-5 w-5" />
                Resume Upload
              </h3>
              <motion.div
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsDragging(false);
                  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                    const filesArray = Array.from(e.dataTransfer.files);
                    if (filesArray.length === 1) {
                      handleFileUpload(filesArray[0]);
                    } else {
                      handleBulkFileUploads(filesArray);
                    }
                  }
                }}
                className={`flex h-48 flex-col items-center justify-center rounded-xl border-2 border-dashed transition-all ${
                  isDragging
                    ? 'border-cyan-500 bg-cyan-500/10 shadow-[0_0_25px_rgba(6,182,212,0.3)]'
                    : 'border-white/20 bg-slate-950/30 hover:border-cyan-500/50 hover:bg-slate-950/50'
                }`}
              >
                <input 
                  type="file" 
                  className="hidden" 
                  id="resume-upload" 
                  multiple
                  onChange={(e) => {
                    if (e.target.files && e.target.files.length > 0) {
                      const filesArray = Array.from(e.target.files);
                      if (filesArray.length === 1) {
                        handleFileUpload(filesArray[0]);
                      } else {
                        handleBulkFileUploads(filesArray);
                      }
                    }
                  }}
                  accept=".pdf,.png,.jpg,.docx"
                />
                <label 
                  htmlFor="resume-upload"
                  className="flex h-full w-full cursor-pointer flex-col items-center justify-center"
                >
                  <Upload className={`mb-3 h-12 w-12 transition-colors ${isDragging ? 'text-cyan-400' : isUploading ? 'text-cyan-500 animate-bounce' : 'text-slate-500'}`} />
                  <p className="text-sm font-medium text-slate-300">
                    {isUploading ? 'Parsing with Groq LLM...' : 'Drag & Drop Resumes'}
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    {isUploading ? 'Please wait 3s...' : 'or click to browse (PDF, PNG, DOCX)'}
                  </p>
                </label>
              </motion.div>
              <p className="mt-3 text-xs text-slate-400 text-center leading-relaxed">
                💡 <span className="text-cyan-400 font-medium">Tip:</span> Upload digital PDFs for maximum accuracy. Images and scans require OCR, which can cause slight score variations.
              </p>
            </div>

            {/* AI Ranking Controller Card */}
            <div className="group rounded-2xl border border-white/10 bg-slate-900/40 p-6 shadow-2xl backdrop-blur-xl transition-all hover:border-purple-500/30 hover:shadow-[0_0_30px_rgba(168,85,247,0.15)]">
              <div className="mb-4 flex items-center justify-between">
                <h3 className="flex items-center gap-2 text-purple-300 font-semibold text-sm">
                  <TrendingUp className="h-5 w-5" />
                  AI Ranking Controller
                </h3>
                <button
                  onClick={resetWeights}
                  className="rounded-lg border border-purple-500/30 bg-purple-500/10 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider text-purple-300 transition-all hover:bg-purple-500/30 hover:text-white"
                >
                  Reset
                </button>
              </div>
              
              {/* Group 1: General Match Priorities */}
              <div className="space-y-4 border-b border-white/5 pb-4 mb-4">
                <div className="flex justify-between text-xs font-semibold text-slate-400">
                  <span>Compatibility vs. Profile Strength</span>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs text-slate-300">
                    <span>Role Fit (JD Match)</span>
                    <span className="font-bold text-purple-400">{Math.round(wMatch * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={Math.round(wMatch * 100)}
                    onChange={(e) => {
                      const val = Number(e.target.value) / 100;
                      setWMatch(val);
                      setWStrength(1.0 - val);
                    }}
                    style={{
                      background: `linear-gradient(to right, rgb(168, 85, 247) 0%, rgb(168, 85, 247) ${Math.round(wMatch * 100)}%, rgb(9, 13, 22) ${Math.round(wMatch * 100)}%, rgb(9, 13, 22) 100%)`
                    }}
                    className="w-full accent-purple-500 h-1.5 rounded-lg appearance-none cursor-pointer"
                  />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs text-slate-300">
                    <span>Candidate General Quality</span>
                    <span className="font-bold text-cyan-400">{Math.round(wStrength * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={Math.round(wStrength * 100)}
                    onChange={(e) => {
                      const val = Number(e.target.value) / 100;
                      setWStrength(val);
                      setWMatch(1.0 - val);
                    }}
                    style={{
                      background: `linear-gradient(to right, rgb(6, 182, 212) 0%, rgb(6, 182, 212) ${Math.round(wStrength * 100)}%, rgb(9, 13, 22) ${Math.round(wStrength * 100)}%, rgb(9, 13, 22) 100%)`
                    }}
                    className="w-full accent-cyan-500 h-1.5 rounded-lg appearance-none cursor-pointer"
                  />
                </div>
              </div>

              {/* Group 2: Match Component Sub-Weights */}
              <div className="space-y-4">
                <div className="flex justify-between text-xs font-semibold text-slate-400">
                  <span>Match Criteria Weights (Must sum to 100%)</span>
                </div>
                
                {/* Semantic */}
                <div className="space-y-2">
                  <div className="flex justify-between text-xs text-slate-300">
                    <span>Semantic Similarity</span>
                    <span className="font-bold text-purple-400">{Math.round(wSem * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={Math.round(wSem * 100)}
                    onChange={(e) => handleSemChange(Number(e.target.value))}
                    style={{
                      background: `linear-gradient(to right, rgb(168, 85, 247) 0%, rgb(168, 85, 247) ${Math.round(wSem * 100)}%, rgb(9, 13, 22) ${Math.round(wSem * 100)}%, rgb(9, 13, 22) 100%)`
                    }}
                    className="w-full accent-purple-500 h-1.5 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                {/* Keyword */}
                <div className="space-y-2">
                  <div className="flex justify-between text-xs text-slate-300">
                    <span>Keyword Overlap</span>
                    <span className="font-bold text-cyan-400">{Math.round(wKw * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={Math.round(wKw * 100)}
                    onChange={(e) => handleKwChange(Number(e.target.value))}
                    style={{
                      background: `linear-gradient(to right, rgb(6, 182, 212) 0%, rgb(6, 182, 212) ${Math.round(wKw * 100)}%, rgb(9, 13, 22) ${Math.round(wKw * 100)}%, rgb(9, 13, 22) 100%)`
                    }}
                    className="w-full accent-cyan-500 h-1.5 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                {/* Experience */}
                <div className="space-y-2">
                  <div className="flex justify-between text-xs text-slate-300">
                    <span>Experience Relevance</span>
                    <span className="font-bold text-purple-400">{Math.round(wExp * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={Math.round(wExp * 100)}
                    onChange={(e) => handleExpChange(Number(e.target.value))}
                    style={{
                      background: `linear-gradient(to right, rgb(168, 85, 247) 0%, rgb(168, 85, 247) ${Math.round(wExp * 100)}%, rgb(9, 13, 22) ${Math.round(wExp * 100)}%, rgb(9, 13, 22) 100%)`
                    }}
                    className="w-full accent-purple-500 h-1.5 rounded-lg appearance-none cursor-pointer"
                  />
                </div>
              </div>
            </div>
          </motion.div>

          {/* RIGHT PANEL - Leaderboard */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="rounded-2xl border border-white/10 bg-slate-900/40 p-6 shadow-2xl backdrop-blur-xl h-fit"
          >
            <div className="mb-6 flex items-center justify-between">
              <h2 className="flex items-center gap-3 text-lg font-semibold text-slate-200">
                <TrendingUp className="h-6 w-6 text-purple-400" />
                Candidate Leaderboard
              </h2>
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.4, type: 'spring' }}
                className="rounded-full border border-cyan-500/50 bg-cyan-500/10 px-4 py-1.5 text-sm font-medium text-cyan-300 shadow-[0_0_15px_rgba(6,182,212,0.3)]"
              >
                {candidates.length} processed
              </motion.div>
            </div>

            {/* Table wrapper for horizontal scroll protection on smaller screens */}
            <div className="w-full overflow-x-auto">
              <div className="min-w-[800px] pr-2">
                {/* Table Header */}
                <div className="mb-3 grid grid-cols-[60px_1fr_100px_110px_110px_140px] gap-4 border-b border-white/5 pb-3 text-xs uppercase tracking-wider text-slate-400 font-semibold">
                  <div>Rank</div>
                  <div>Candidate Name</div>
                  <div>AI Score</div>
                  <div>Semantic</div>
                  <div>ML Strength</div>
                  <div>Decision</div>
                </div>

                {/* Candidate Rows */}
                <div className="space-y-3">
                  {candidates.length === 0 ? (
                    <div className="py-12 text-center text-slate-500 text-sm">
                      No candidates yet. Set a JD and upload a resume!
                    </div>
                  ) : (
                    candidates.map((candidate, index) => (
                      <motion.div
                        key={candidate.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 + index * 0.05 }}
                        className="group grid grid-cols-[60px_1fr_100px_110px_110px_140px] gap-4 rounded-xl border border-white/5 bg-slate-950/50 p-4 backdrop-blur-sm transition-all hover:border-purple-500/30 hover:bg-slate-900/60 hover:shadow-[0_0_20px_rgba(168,85,247,0.1)]"
                      >
                        {/* Rank */}
                        <div className="flex items-center">
                          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-purple-500/10 text-sm font-bold text-purple-400">
                            {index + 1}
                          </div>
                        </div>

                        {/* Name */}
                        <div className="flex items-center gap-2.5 font-medium text-slate-200 relative group/audit">
                          {isBlindMode && !revealedIds.includes(candidate.id) ? (
                            <div className="flex items-center gap-2">
                              <span className="text-slate-400 italic text-sm">Candidate #{candidate.id}</span>
                              <button
                                onClick={() => {
                                  setRevealedIds([...revealedIds, candidate.id]);
                                  toast.success(`Identity of Candidate #${candidate.id} revealed!`);
                                }}
                                className="rounded-lg border border-purple-500/30 bg-purple-500/10 px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider text-purple-300 transition-all hover:bg-purple-500/25 hover:text-white"
                              >
                                Reveal
                              </button>
                            </div>
                          ) : (
                            <>
                              <span>{candidate.name}</span>
                              {candidate.github_username ? (
                                <button
                                  onClick={() => handleOpenGithubModal(candidate)}
                                  className="text-purple-400 hover:text-purple-300 transition-colors hover:scale-110 flex items-center"
                                  title={`GitHub handle: @${candidate.github_username}`}
                                >
                                  <Github className="h-3.5 w-3.5 drop-shadow-[0_0_8px_rgba(168,85,247,0.6)]" />
                                </button>
                              ) : (
                                <button
                                  onClick={() => handleOpenGithubModal(candidate)}
                                  className="text-slate-600 hover:text-slate-400 transition-colors flex items-center"
                                  title="Link GitHub Handle"
                                >
                                  <Github className="h-3.5 w-3.5 opacity-30 hover:opacity-100" />
                                </button>
                              )}
                            </>
                          )}
                          {candidate.audit_status?.flagged && (
                            <div className="relative cursor-pointer">
                              <span className="text-amber-500 animate-pulse text-sm">⚠️</span>
                              {/* Hover Tooltip */}
                              <div className="absolute left-6 top-1/2 -translate-y-1/2 hidden group-hover/audit:block z-50 w-72 rounded-xl border border-amber-500/20 bg-slate-950/95 p-3 shadow-2xl backdrop-blur-md">
                                <p className="text-xs font-bold text-amber-400 mb-1 flex items-center gap-1.5">
                                  <span>Optimization Warning</span>
                                </p>
                                <ul className="space-y-1 list-disc pl-3 text-[10px] leading-relaxed text-slate-300 font-normal">
                                  {candidate.audit_status.warnings.map((w, idx) => (
                                    <li key={idx}>{w}</li>
                                  ))}
                                </ul>
                              </div>
                            </div>
                          )}
                        </div>

                        {/* AI Score with sparkline */}
                        <div className="flex flex-col justify-center">
                          <div className="mb-1 text-lg font-bold text-purple-300">
                            {candidate.score}%
                          </div>
                          <div className="h-1.5 overflow-hidden rounded-full bg-slate-800 w-full">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${candidate.score}%` }}
                              transition={{ delay: 0.3 + index * 0.05, duration: 0.8 }}
                              className="h-full rounded-full bg-gradient-to-r from-purple-500 to-cyan-500"
                            />
                          </div>
                        </div>

                        {/* Semantic Match */}
                        <div className="flex items-center text-sm font-medium text-slate-400">
                          {candidate.semanticMatch}%
                        </div>

                        {/* ML Strength */}
                        <div className="flex items-center text-sm font-medium text-slate-400">
                          {candidate.mlStrength}%
                        </div>

                        {/* Decision Badge */}
                        <div className="flex items-center">
                          <div
                            className={`rounded-full border px-3 py-1.5 text-xs font-bold uppercase tracking-wider transition-all ${getDecisionStyle(
                              candidate.decision
                            )}`}
                          >
                            {candidate.decision}
                          </div>
                        </div>
                      </motion.div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Chat Sidebar Drawer */}
      <motion.div
        initial={{ x: '100%' }}
        animate={{ x: isChatOpen ? 0 : '100%' }}
        transition={{ type: 'spring', damping: 25, stiffness: 200 }}
        className="fixed inset-y-0 right-0 z-50 w-full max-w-[450px] border-l border-white/10 bg-slate-950/80 p-6 shadow-2xl backdrop-blur-2xl flex flex-col"
      >
        {/* Sidebar Header */}
        <div className="flex items-center justify-between border-b border-white/5 pb-4 mb-4">
          <div className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-purple-400" />
            <h3 className="font-bold text-slate-200">Recruiter AI Assistant</h3>
          </div>
          <button
            onClick={() => setIsChatOpen(false)}
            className="rounded-lg p-1 text-slate-400 hover:bg-white/5 hover:text-slate-200 transition-colors"
          >
            <span className="text-xl leading-none">&times;</span>
          </button>
        </div>

        {/* Chat Messages Feed */}
        <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-1 scrollbar-thin scrollbar-thumb-white/10">
          {chatMessages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center p-6 text-slate-500">
              <Brain className="h-10 w-10 text-slate-600 mb-3 animate-pulse" />
              <p className="text-sm font-medium">Ask me about the candidate database!</p>
              <p className="text-xs text-slate-600 mt-1 max-w-[280px]">
                Try asking: "Who has python skills?" or "Compare the experience level of the top 3 candidates."
              </p>
            </div>
          ) : (
            chatMessages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex flex-col ${
                  msg.role === 'user' ? 'items-end' : 'items-start'
                }`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                    msg.role === 'user'
                      ? 'bg-purple-600/90 text-white rounded-tr-none'
                      : 'bg-slate-900/80 border border-white/5 text-slate-300 rounded-tl-none'
                  }`}
                >
                  <div className="whitespace-pre-wrap">{msg.content}</div>
                </div>
              </div>
            ))
          )}
          {isSendingChat && (
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <div className="h-1.5 w-1.5 rounded-full bg-cyan-400 animate-bounce" />
              <div className="h-1.5 w-1.5 rounded-full bg-cyan-400 animate-bounce [animation-delay:0.2s]" />
              <div className="h-1.5 w-1.5 rounded-full bg-cyan-400 animate-bounce [animation-delay:0.4s]" />
              <span>Thinking...</span>
            </div>
          )}
        </div>

        {/* Chat Input form */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSendChatMessage();
          }}
          className="flex gap-2"
        >
          <input
            type="text"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            placeholder="Ask a question..."
            className="flex-1 rounded-xl border border-white/5 bg-slate-950/60 px-4 py-2 text-sm text-slate-200 placeholder-slate-500 outline-none ring-purple-500/50 transition-all focus:border-purple-500/50 focus:ring-2"
            disabled={isSendingChat}
          />
          <button
            type="submit"
            disabled={isSendingChat || !chatInput.trim()}
            className="rounded-xl border border-purple-500/50 bg-gradient-to-r from-purple-600 to-cyan-600 px-4 py-2 text-sm font-medium text-white hover:border-purple-400 hover:shadow-[0_0_15px_rgba(168,85,247,0.3)] disabled:opacity-50 transition-all"
          >
            Send
          </button>
        </form>
      </motion.div>

      {/* Overlay when Chat Sidebar is Open */}
      {isChatOpen && (
        <div
          onClick={() => setIsChatOpen(false)}
          className="fixed inset-0 z-40 bg-black/40 backdrop-blur-sm transition-all"
        />
      )}

      {/* GitHub Developer Profile Modal */}
      {isGithubModalOpen && activeGithubCandidate && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="w-full max-w-[500px] rounded-2xl border border-white/10 bg-slate-950 p-6 shadow-2xl backdrop-blur-2xl"
          >
            {/* Modal Header */}
            <div className="mb-4 flex items-center justify-between border-b border-white/5 pb-3">
              <div className="flex items-center gap-2">
                <Github className="h-6 w-6 text-purple-400" />
                <div>
                  <h3 className="font-bold text-slate-200">GitHub Developer Profile</h3>
                  <p className="text-[10px] text-slate-500 font-medium">For candidate: {activeGithubCandidate.name}</p>
                </div>
              </div>
              <button
                onClick={() => setIsGithubModalOpen(false)}
                className="rounded-lg p-1 text-slate-400 hover:bg-white/5 hover:text-slate-200"
              >
                <span className="text-xl leading-none">&times;</span>
              </button>
            </div>

            {/* Link Handle form */}
            {(!activeGithubCandidate.github_username || isLinkingGithub) ? (
              <form onSubmit={handleLinkGithubSubmit} className="space-y-4 py-4">
                <div className="space-y-2">
                  <label className="text-xs text-slate-400 font-medium">Enter candidate's GitHub handle:</label>
                  <input
                    type="text"
                    value={manualGithubInput}
                    onChange={(e) => setManualGithubInput(e.target.value)}
                    placeholder="e.g. octocat"
                    className="w-full rounded-xl border border-white/5 bg-slate-900/60 px-4 py-2 text-sm text-slate-200 placeholder-slate-500 outline-none ring-purple-500/50 focus:border-purple-500/50 focus:ring-2"
                    disabled={isLinkingGithub}
                  />
                </div>
                <div className="flex gap-2 justify-end">
                  <button
                    type="button"
                    onClick={() => {
                      if (activeGithubCandidate.github_username) {
                        setManualGithubInput(activeGithubCandidate.github_username);
                      } else {
                        setIsGithubModalOpen(false);
                      }
                    }}
                    className="rounded-xl border border-white/5 bg-white/5 px-4 py-2 text-xs font-semibold text-slate-300 hover:bg-white/10"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="rounded-xl border border-purple-500/50 bg-gradient-to-r from-purple-600 to-cyan-600 px-4 py-2 text-xs font-semibold text-white hover:border-purple-400 hover:shadow-[0_0_15px_rgba(168,85,247,0.3)]"
                    disabled={isLinkingGithub}
                  >
                    {isLinkingGithub ? 'Saving...' : 'Link Profile'}
                  </button>
                </div>
              </form>
            ) : (
              // Main details display
              <div className="space-y-6 py-2">
                {githubModalLoading ? (
                  <div className="py-12 flex flex-col items-center justify-center space-y-3">
                    <div className="h-6 w-6 animate-spin rounded-full border-2 border-purple-500 border-t-transparent" />
                    <p className="text-xs text-slate-500 font-medium">Querying public GitHub API...</p>
                  </div>
                ) : !githubStats ? (
                  <div className="py-8 text-center text-slate-500 text-sm">
                    <p className="font-medium">No GitHub statistics found for handle: <span className="text-purple-400 font-bold">@{activeGithubCandidate.github_username}</span></p>
                    <div className="mt-4 flex justify-center gap-2">
                      <button
                        onClick={() => {
                          setManualGithubInput('');
                          const updated = { ...activeGithubCandidate, github_username: null };
                          setActiveGithubCandidate(updated);
                        }}
                        className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-300 hover:bg-white/10"
                      >
                        Change Username
                      </button>
                      <button
                        onClick={() => handleOpenGithubModal(activeGithubCandidate)}
                        className="rounded-lg border border-purple-500/30 bg-purple-500/10 px-3 py-1.5 text-xs text-purple-300 hover:bg-purple-500/20"
                      >
                        Retry
                      </button>
                    </div>
                  </div>
                ) : githubStats.rate_limit_exceeded ? (
                  <div className="py-8 text-center text-slate-500 text-sm">
                    <p className="text-amber-500 font-medium mb-1">⚠️ GitHub API Rate Limit Exceeded</p>
                    <p className="text-xs leading-relaxed max-w-[320px] mx-auto text-slate-500">
                      Unauthenticated developer requests are limited by GitHub. To prevent this, link a personal token in your env or try again later.
                    </p>
                    <p className="mt-4 text-xs font-medium">Target profile handle: <span className="text-purple-400 font-semibold">@{activeGithubCandidate.github_username}</span></p>
                    <div className="mt-4 flex justify-center gap-2">
                      <button
                        onClick={() => {
                          setManualGithubInput('');
                          const updated = { ...activeGithubCandidate, github_username: null };
                          setActiveGithubCandidate(updated);
                        }}
                        className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-300 hover:bg-white/10"
                      >
                        Change Username
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    {/* Repository Counts Stats Grid */}
                    <div className="grid grid-cols-3 gap-4 text-center">
                      <div className="rounded-xl border border-white/5 bg-slate-900/40 p-3">
                        <p className="text-lg font-bold text-slate-200">{githubStats.repos_count}</p>
                        <p className="text-[10px] text-slate-500">Public Repos</p>
                      </div>
                      <div className="rounded-xl border border-white/5 bg-slate-900/40 p-3">
                        <p className="text-lg font-bold text-purple-400">{githubStats.stars_count}</p>
                        <p className="text-[10px] text-slate-500">Total Stars</p>
                      </div>
                      <div className="rounded-xl border border-white/5 bg-slate-900/40 p-3">
                        <p className="text-lg font-bold text-cyan-400">{githubStats.forks_count}</p>
                        <p className="text-[10px] text-slate-500">Total Forks</p>
                      </div>
                    </div>

                    {/* Language Usage Stack Chart */}
                    <div className="space-y-3">
                      <h4 className="text-xs font-semibold text-slate-400">Primary Languages Stack (by Repository size)</h4>
                      {Object.keys(githubStats.languages).length === 0 ? (
                        <p className="text-xs text-slate-500 italic">No language data found in public repositories.</p>
                      ) : (
                        <div className="space-y-2.5">
                          {Object.entries(githubStats.languages).map(([lang, pct]: any) => (
                            <div key={lang} className="space-y-1">
                              <div className="flex justify-between text-xs text-slate-300">
                                <span>{lang}</span>
                                <span className="font-bold text-purple-400">{pct}%</span>
                              </div>
                              <div className="h-1.5 overflow-hidden rounded-full bg-slate-900 w-full">
                                <div
                                  style={{ width: `${pct}%` }}
                                  className="h-full rounded-full bg-gradient-to-r from-purple-500 to-cyan-500"
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Action buttons */}
                    <div className="flex justify-between border-t border-white/5 pt-4 mt-2">
                      <button
                        onClick={() => {
                          setManualGithubInput(activeGithubCandidate.github_username || '');
                          const updated = { ...activeGithubCandidate, github_username: null };
                          setActiveGithubCandidate(updated);
                        }}
                        className="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-xs font-semibold text-slate-300 hover:bg-white/10"
                      >
                        Modify Handle
                      </button>
                      <button
                        onClick={() => handleOpenGithubModal(activeGithubCandidate)}
                        className="rounded-xl border border-purple-500/30 bg-purple-500/10 px-4 py-2 text-xs font-semibold text-purple-300 hover:bg-purple-500/20"
                      >
                        Reload
                      </button>
                    </div>
                  </>
                )}
              </div>
            )}
          </motion.div>
        </div>
      )}
    </div>
  );
}
