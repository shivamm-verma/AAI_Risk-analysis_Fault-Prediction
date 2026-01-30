import React from 'react'

export default function Header() {
  return (
    <div className="bg-white">
      {/* Top Section */}
      <div className="max-w-7xl mx-auto px-4 py-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-gray-200">

        {/* Logo + Title */}
        <div className="flex items-center gap-3">
          <div className="h-14 w-14 sm:h-16 sm:w-16 bg-blue-900 rounded flex items-center justify-center shrink-0">
            <img src="/logo.png" alt="AAI Logo" className="h-full w-full object-cover rounded" />
          </div>
          <div>
            <h1 className="text-sm sm:text-lg font-bold text-blue-900">
              AIRPORTS AUTHORITY OF INDIA
            </h1>
            <p className="text-[10px] sm:text-xs text-gray-600">
              (A Miniratna - Category -1 Public Sector Enterprise)
            </p>
          </div>
        </div>

        {/* Buttons */}
        <div className="flex flex-wrap gap-2 justify-start sm:justify-end">
          <button className="px-3 py-2 bg-blue-900 text-white text-xs sm:text-sm font-semibold rounded hover:bg-blue-800">
            Pass AI Animation
          </button>
          <button className="px-3 py-2 bg-blue-900 text-white text-xs sm:text-sm font-semibold rounded hover:bg-blue-800">
            Emp Login
          </button>
          <button className="px-3 py-2 bg-blue-900 text-white rounded hover:bg-blue-800">
            📺
          </button>
          <button className="px-3 py-2 bg-blue-900 text-white rounded hover:bg-blue-800 font-bold text-xs sm:text-sm">
            हि
          </button>
        </div>
      </div>

      {/* Announcement Bar */}
      <div className="bg-red-600 text-white py-2 px-3 text-center text-[10px] sm:text-sm leading-snug">
        <span className="font-semibold block sm:inline">
          Schedule for recruitment through GATE 2023, GATE 2024, GATE 2025
        </span>
        <span className="block sm:inline sm:ml-2">
          Advertisement for Hiring of Co-Pilots (P2)-B350 B360 on contract basis at Flight Inspection
        </span>
      </div>
    </div>
  )
}
