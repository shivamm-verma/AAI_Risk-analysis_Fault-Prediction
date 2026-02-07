import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import { ChevronDown, Home, Info, Briefcase, Settings } from 'lucide-react'

const navigation = [
  { label: 'HOME', href: '/', icon: Home },
  { label: 'ABOUT US', href: '/about', icon: Info },
  { label: 'BUSINESS INFO', href: '#', icon: Briefcase },
  { label: 'SERVICES', href: '#', icon: Settings },
  { label: 'TENDER', href: '#', icon: Settings },
  { label: 'VIGILANCE', href: '#', icon: Settings },
  { label: 'PUBLIC INFO', href: '#', icon: Info },
  { label: 'CSR', href: '#', icon: Info },
  { label: 'AAI AIRPORTS', href: '#', icon: Home },
  { label: 'TOOLS', href: '/tools', icon: Settings },
]

export default function Navbar() {
  const [open, setOpen] = useState(false)

  return (
    <nav className="bg-blue-900 text-white">
      <div className="max-w-7xl mx-auto px-4">

        {/* Desktop */}
        <ul className="hidden sm:flex justify-around">
          {navigation.map(({ label, href }) => (
            <li key={label}>
              <Link
                to={href}
                className="px-4 py-3 text-sm font-semibold hover:bg-blue-800 block"
              >
                {label}
              </Link>
            </li>
          ))}
        </ul>

        {/* Mobile Dropdown */}
        <div className="sm:hidden relative">
          <button
            onClick={() => setOpen(!open)}
            className="w-full flex items-center justify-between px-4 py-3 font-semibold"
          >
            MENU
            <ChevronDown className={`w-4 h-4 transition ${open ? 'rotate-180' : ''}`} />
          </button>

          {open && (
            <ul className="absolute left-0 right-0 bg-blue-800 z-50">
              {navigation.map(({ label, href, icon: Icon }) => (
                <li key={label}>
                  <Link
                    to={href}
                    className="flex items-center gap-2 px-4 py-3 text-sm hover:bg-blue-700"
                    onClick={() => setOpen(false)}
                  >
                    <Icon size={16} />
                    {label}
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </div>

      </div>
    </nav>
  )
}
