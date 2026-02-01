import React from 'react'

import { ChevronLeft, ChevronRight } from 'lucide-react';
import { useState } from 'react';

export default function Home() {
  const [currentSlide, setCurrentSlide] = useState(0);

  const slides = [
    {
      title: 'New Year, New Horizons',
      subtitle: 'AAI begins the New Year with a Commitment to Service, Safety and Excellence',
      image: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      highlights: true,
    },
  ];

  return (
    <div className="bg-linear-to-r from-blue-400 to-blue-800 text-white py-12">
      <div className="max-w-7xl mx-auto px-4">
        {/* Main Carousel */}
        <div
          className="relative rounded-lg overflow-hidden h-96 flex items-center justify-center mb-8"
          style={{
            backgroundImage: 'url("/banner.jpg")',
            backgroundSize: 'cover',
            backgroundPosition: 'center',
          }}
        >
          <div className="absolute inset-0 bg-black/40"></div>
          <div className="relative z-10 text-center">
            <h2 className="text-5xl font-bold mb-4 text-yellow-300">{slides[currentSlide].title}</h2>
            <p className="text-xl">{slides[currentSlide].subtitle}</p>
          </div>

          <button
            onClick={() => setCurrentSlide((currentSlide - 1 + slides.length) % slides.length)}
            className="absolute left-4 z-20 bg-white/30 hover:bg-white/50 p-2 rounded transition"
          >
            <ChevronLeft className="w-6 h-6" />
          </button>
          <button
            onClick={() => setCurrentSlide((currentSlide + 1) % slides.length)}
            className="absolute right-4 z-20 bg-white/30 hover:bg-white/50 p-2 rounded transition"
          >
            <ChevronRight className="w-6 h-6" />
          </button>
        </div>

        {/* Highlights of 2025 */}
        <div className="text-center mb-8">
          <h3 className="text-xl font-semibold mb-6">Highlights of 2025</h3>
          <div className="grid grid-cols-5 gap-4">
            <div className="bg-blue-700 p-4 rounded cursor-pointer hover:bg-blue-600 transition">
              <div className="text-sm font-semibold">Digi Yatra</div>
              <div className="text-xs mt-2">Biometric at 14 AAI Airports</div>
            </div>
            <div className="bg-blue-700 p-4 rounded cursor-pointer hover:bg-blue-600 transition">
              <div className="text-sm font-semibold">Cargo of</div>
              <div className="text-xs mt-2">Excellence</div>
            </div>
            <div className="bg-blue-700 p-4 rounded cursor-pointer hover:bg-blue-600 transition">
              <div className="text-sm font-semibold">Union Vixi Cam</div>
              <div className="text-xs mt-2">24/7 Services</div>
            </div>
            <div className="bg-blue-700 p-4 rounded cursor-pointer hover:bg-blue-600 transition">
              <div className="text-sm font-semibold">IOS operations</div>
              <div className="text-xs mt-2">World Class</div>
            </div>
            <div className="bg-green-600 p-4 rounded cursor-pointer hover:bg-green-500 transition">
              <div className="text-sm font-semibold">SUGAM</div>
              <div className="text-xs mt-2">Green Services</div>
            </div>
          </div>
          <button className="mt-6 text-sm font-semibold hover:underline">View more about AAI Important Notices</button>
        </div>
      </div>
    </div>
  );
}

