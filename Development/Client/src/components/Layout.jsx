import React from 'react'
import { Outlet } from 'react-router-dom'
import Navbar from './Navbar'
import Footer from './Footer'
import Header from './Header'

function Layout() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Header at top */}
      <Header />

      {/* Navbar*/}
      <Navbar />
      
      {/* Main content area - grows to fill available space */}
      <main className="grow">
        <Outlet />
      </main>
      
      {/* Footer at bottom */}
      <Footer />
    </div>
  )
}

export default Layout
