"use client";
import React, { useState } from 'react';
import Image from 'next/image';

export default function ParkingAnalyzer() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [emptySlots, setEmptySlots] = useState([]);
  const [emptySlotPositions, setEmptySlotPositions] = useState([]);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({
    totalSpots: 0,
    totalEmpty: 0,
    totalOccupied: 0,
  });
  
  // API URL - change to match your Flask server port
  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5001';
  
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setError(null);
      setProcessedImage(null);
      setEmptySlots([]);
      
      // Create preview URL
      const reader = new FileReader();
      reader.onload = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };
  
  const analyzeImage = async () => {
    if (!selectedImage) {
      setError("Please select an image first");
      return;
    }
    
    setAnalyzing(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('image', selectedImage);
      
      const response = await fetch(`${API_URL}/api/detect`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }
      
      // Set the results
      if (data.image_path) {
        setProcessedImage(`${API_URL}/${data.image_path}`);
      }
      
      // Check the structure of empty_spots
      if (Array.isArray(data.empty_spots)) {
        // If empty_spots is already an array of objects with section info
        if (data.empty_spots.length > 0 && data.empty_spots[0].section) {
          // Group by section
          const groupedBySection = data.empty_spots.reduce((acc, spot) => {
            const section = spot.section;
            if (!acc[section]) {
              acc[section] = [];
            }
            acc[section].push(spot);
            return acc;
          }, {});
          
          setEmptySlots(Object.entries(groupedBySection).map(([section, slots]) => ({
            section,
            slots
          })));
        } else {
          // If empty_spots is an array of strings (spot IDs)
          const groupedSlots = data.empty_spots.reduce((acc, spot) => {
            // For simple string values, get the first character as section
            const spotId = typeof spot === 'string' ? spot : spot.id;
            const section = spotId.substring(0, 1);
            
            if (!acc[section]) {
              acc[section] = [];
            }
            acc[section].push({
              id: spotId,
              section: section,
              status: 'empty'
            });
            return acc;
          }, {});
          
          setEmptySlots(Object.entries(groupedSlots).map(([section, slots]) => ({
            section,
            slots
          })));
        }
      } else {
        setEmptySlots([]);
      }
      
      // Set statistics
      setStats({
        totalSpots: data.total_spots || 0,
        totalEmpty: data.available_spots || 0,
        totalOccupied: (data.total_spots || 0) - (data.available_spots || 0)
      });
      
      // For demonstration, set random positions for empty slot markers
      // In a real app, you would use actual coordinates from the backend
      if (Array.isArray(data.empty_spots)) {
        const spotIds = data.empty_spots.map(spot => 
          typeof spot === 'string' ? spot : spot.id
        );
        
        setEmptySlotPositions(
          spotIds.slice(0, 5).map((spot) => ({
            id: spot,
            top: `${Math.floor(Math.random() * 80) + 10}%`,
            left: `${Math.floor(Math.random() * 80) + 10}%`
          }))
        );
      }
      
    } catch (err) {
      console.error("Analysis error:", err);
      setError(err.message || "Failed to analyze image. Please try again.");
    } finally {
      setAnalyzing(false);
    }
  };
  
  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-12">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-indigo-900 mb-2">Smart Parking Analyzer</h1>
          <p className="text-lg text-indigo-700">Upload a parking lot image to find available parking spots</p>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-2 bg-white p-6 rounded-xl shadow-lg">
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">Upload Parking Lot Image</label>
              <input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
                id="image-upload"
              />
              <label
                htmlFor="image-upload"
                className="flex justify-center items-center w-full h-40 px-4 transition bg-white border-2 border-gray-300 border-dashed rounded-md appearance-none cursor-pointer hover:border-indigo-600 focus:outline-none"
              >
                <span className="flex items-center space-x-2">
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  <span className="font-medium text-gray-600">
                    Drop files to upload, or
                    <span className="text-indigo-600 underline ml-1">browse</span>
                  </span>
                </span>
              </label>
            </div>
            
            {error && (
              <div className="mt-4 p-3 bg-red-100 text-red-700 rounded-md">
                {error}
              </div>
            )}
            
            {imagePreview && (
              <div className="mt-6">
                <div className="relative w-full h-96 rounded-lg overflow-hidden">
                  <img 
                    src={processedImage || imagePreview}
                    alt="Parking lot preview" 
                    className="object-cover w-full h-full"
                  />
                  
                  {/* Image overlay markers for empty spots */}
                  {emptySlotPositions.length > 0 && !analyzing && (
                    <>
                    
                    </>
                  )}
                </div>
                
                <div className="mt-4 flex justify-center">
                  <button
                    onClick={analyzeImage}
                    disabled={analyzing || !selectedImage}
                    className="px-6 py-3 bg-indigo-600 text-white font-medium rounded-lg shadow-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-opacity-50 disabled:opacity-50"
                  >
                    {analyzing ? (
                      <span className="flex items-center">
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Analyzing...
                      </span>
                    ) : (
                      'Analyze Parking Spaces'
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>
          
          {/* Results Section */}
          <div className="bg-white p-6 rounded-xl shadow-lg">
            <h2 className="text-xl font-bold text-indigo-900 mb-4">Available Parking Spots</h2>
            
            {emptySlots.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-64 text-gray-500">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p>Upload and analyze an image to see available spots</p>
              </div>
            ) : (
              <div className="space-y-4 max-h-96 overflow-y-auto pr-2">
                {emptySlots.map(({ section, slots }) => (
                  <div key={section} className="bg-indigo-50 rounded-lg p-4">
                    <h3 className="font-bold text-indigo-800 mb-2">Section {section}</h3>
                    <div className="grid grid-cols-3 gap-2">
                      {slots.map(slot => (
                        <div 
                          key={slot.id} 
                          className="bg-green-100 border border-green-300 rounded-md py-2 text-center text-green-800 font-medium"
                        >
                          {slot.id}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
            
            {stats.totalSpots > 0 && (
              <div className="mt-6 pt-4 border-t border-gray-200">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-medium text-gray-700">Total Available:</span>
                  <span className="text-xl font-bold text-indigo-600">{stats.totalEmpty}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div 
                    className="bg-indigo-600 h-2.5 rounded-full" 
                    style={{ width: `${(stats.totalEmpty / stats.totalSpots) * 100}%` }}
                  ></div>
                </div>
                <p className="text-sm text-gray-500 mt-1">
                  {stats.totalEmpty} out of {stats.totalSpots} spots available
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}