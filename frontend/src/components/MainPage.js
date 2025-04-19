import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { jwtDecode } from "jwt-decode";
import "./MainPage.css";
import Navbar from "./Navbar";
import fetchFramework from "../framework/fetchFramework";
import { useRef } from "react";
import Spectrogram from "../Spectrogram.png"

const MainPage = () => {
  const audioRef = useRef(null);
  const navigate = useNavigate();
  const [currentTime, setCurrentTime] = useState(0);
  const [selectedModel, setSelectedModel] = useState("EEND-EDA");
  const [selectedSpeakers, setSelectedSpeakers] = useState("2 Speakers");
  const [selectedFile, setSelectedFile] = useState(null);
  const [diarizationResults, setDiarizationResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [duration, setDuration] = useState(1); // default to prevent divide by zero
  const [image, setImage] = useState();

  const handleLogout = () => {
    localStorage.removeItem("access");
    localStorage.removeItem("refresh");
    navigate("/login");
  };

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleSubmit = async(event) => {
    setIsLoading(true);
    const formData = new FormData();
    formData.append("model", selectedModel);
    formData.append("spk", selectedSpeakers);
    formData.append("audio", selectedFile);
    
    event.preventDefault();
    try {
      let response = await fetchFramework({endpoint: "/api/speaker_diarization/", form: formData});
      console.log(response.diarization_result);
      setDiarizationResults(response.diarization_result);
      setImage(response.image)
    }
    catch (error) {
      console.error(error);
    }finally {
      setIsLoading(false);
    }
  };
 
  const speakerColors = {
    'speaker_0': '#8884d8',
    'speaker_1': '#82ca9d',
    'speaker_2': '#ffc658',
    'speaker_3': '#ff8042'
  };

  return (
    <div className="second-container font-play">
      <Navbar/>
      
      <div className="main-content pb-32"> {/* Added bottom padding */}
      
        <h1 className="text-yellow-500 text-3xl font-bold center-heading">Speakers Diarization</h1>
        
        <div className="flex items-center justify-center bg-gradient-to-r mb-10">
          <div className="p-6 w-96 bg-white/10 backdrop-blur-lg shadow-lg rounded-2xl border border-white/20">
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* File Upload Section */}
            <div className="flex flex-col items-center gap-3">
              <label className="text-white text-lg font-semibold">
                Upload Audio File
              </label>
              <input
                type="file"
                accept="audio/*"
                onChange={handleFileChange}
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="cursor-pointer bg-white/20 text-white py-2 px-4 rounded-lg transition-all duration-300 hover:bg-white/30"
              >
                Choose File
              </label>
              {selectedFile && (
                <p className="text-sm text-gray-300">Selected: {selectedFile.name}</p>
              )}
            </div>

            {/* Model Selection Dropdown */}
            <div className="flex flex-col items-center gap-2">
              <label className="text-white text-lg font-semibold">Select Model</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="bg-white/20 text-white px-4 py-2 rounded-lg w-full backdrop-blur-md focus:outline-none focus:ring-2 focus:ring-blue-300"
              >
                <option value="eend-eda" className="text-black">EEND-EDA</option>
                <option value="diaper"className="text-black">EEND-DiaPer</option>
              </select>
            </div>
            {/* Speaker Selection Dropdown */}
            <div className="flex flex-col items-center gap-2">
              <label className="text-white text-lg font-semibold">Select Speakers</label>
              <select
                value={selectedSpeakers}
                onChange={(e) => setSelectedSpeakers(e.target.value)}
                className="bg-white/20 text-white px-4 py-2 rounded-lg w-full backdrop-blur-md focus:outline-none focus:ring-2 focus:ring-blue-300"
              >
                <option value="2" className="text-black">2 Speakers</option>
                <option value="3"className="text-black">3 Speakers</option>
                <option value="4"className="text-black">4 Speakers</option>
                <option value="M"className="text-black">Mixed Speakers</option>
              </select>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              className="w-full bg-blue-500 text-white font-semibold py-2 px-4 rounded-lg shadow-md transition-all duration-300 hover:bg-blue-600 disabled:opacity-50"
              disabled={!selectedFile || isLoading}
            >
              {isLoading ? (
                <div className="flex items-center justify-center">
                  <div className="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full mr-2"></div>
                  Processing...
                </div>
              ) : (
                'Process Audio'
              )}
            </button>
          </form>
          </div>
        </div>

        {diarizationResults && (
  <div className="results-container">
    <div className="compact-results flex flex-col items-center">
      <h3 className="center-heading">Diarization Output</h3>
      <div className="relative w-full h-50 min-w-[80vw] rounded-md overflow-hidden border border-gray-500 bg-black">
        <img
          src={`data:image/png;base64,${image}`}
          alt="Spectrogram"
          className="w-full h-full object-contain"
        />
      </div>
      <div className="table-wrapper flex justify-center w-full">
        <table className="min-w-[80vw] border-collapse rounded-lg overflow-hidden shadow">
          <thead className="bg-blue-600 text-white">
            <tr>
              <th className="py-3 px-4 text-center">Speaker</th>
              <th className="py-3 px-4 text-center w-1/4">Time</th>
              <th className="py-3 px-4 text-center">Transcript</th>
            </tr>
          </thead>
          <tbody>
            {diarizationResults.map(([speaker, start, end, text], index) => {
              const isActive = currentTime >= start && currentTime <= end;
              
              // Define speaker colors
              const speakerColors = {
                'spk0': 'text-purple-600',
                'spk1': 'text-green-600',
                'spk2': 'text-yellow-600',
                'spk3': 'text-orange-600',
                'spk4': 'text-blue-600',
                'spk5': 'text-red-600',
                // Add more if needed
              };
              
              // Get color for current speaker or default to blue
              const speakerColor = speakerColors[speaker] || 'text-blue-600';

              return (
                <tr
                  key={index}
                  className={`transition ${
                    isActive ? "bg-yellow-100 animate-pulse" : index % 2 === 0 ? "bg-gray-50" : "bg-white"
                  } hover:bg-blue-50`}
                >
                  <td className={`py-3 px-4 font-medium ${speakerColor}`}>
                    <span className={`inline-block w-3 h-3 rounded-full mr-2 ${
                      speaker === 'spk0' ? 'bg-purple-500' :
                      speaker === 'spk1' ? 'bg-green-500' :
                      speaker === 'spk2' ? 'bg-yellow-500' :
                      speaker === 'spk3' ? 'bg-orange-500' :
                      speaker === 'spk4' ? 'bg-blue-500' :
                      'bg-red-500'
                    }`}></span>
                    {speaker}
                  </td>
                  <td className="py-3 px-4 italic text-gray-600">
                    {start.toFixed(2)}s - {end.toFixed(2)}s
                  </td>
                  <td className="py-3 px-4 text-gray-800">{text}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  </div>
)}
      </div>
      
      {/* Fixed Audio Player at Bottom */}
      {selectedFile && diarizationResults && (
        <div className="fixed bottom-0 left-0 right-0 bg-none shadow-lg z-50 py-3 ">
          <div className="max-w-8xl mx-auto px-4">
            <audio
              ref={audioRef}
              controls
              onTimeUpdate={() => setCurrentTime(audioRef.current.currentTime)}
              onLoadedMetadata={() => setDuration(audioRef.current.duration)}
              className="w-full max-w-4xl mx-auto"
            >
              <source src={URL.createObjectURL(selectedFile)} type={selectedFile.type} />
              Your browser does not support the audio element.
            </audio>
          </div>
        </div>
      )}
    </div>
  );  
};

export default MainPage;