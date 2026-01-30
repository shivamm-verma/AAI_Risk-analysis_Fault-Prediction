import React from 'react'

function Footer() {
  return (
    <footer className="bg-blue-900 text-white py-12 overflow-x-hidden">
      <div className="max-w-7xl mx-auto px-4">

        {/* Main Footer */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8 mb-12">

          {/* Important Links */}
          <div className="break-words">
            <h3 className="font-bold text-base mb-4">IMPORTANT LINKS</h3>
            <ul className="space-y-2 text-sm">
              {[
                'AAI Startup Initiative',
                'Other AAI Websites Link',
                'External Links',
                'Union Election',
                'NAC',
                'GST',
                'AAI EDCPS',
                'Arbitration & Mediation',
                'Voluntary Safety Reporting',
                'App Based Cab Aggregator',
                'FTO Slot Portal',
              ].map(item => (
                <li key={item}>
                  <a href="#" className="hover:underline break-words">{item}</a>
                </li>
              ))}
            </ul>
          </div>

          {/* Media */}
          <div className="break-words">
            <h3 className="font-bold text-base mb-4">MEDIA</h3>
            <ul className="space-y-2 text-sm">
              {['Press Releases', 'Latest News', 'Coverage', 'Photo Gallery'].map(item => (
                <li key={item}>
                  <a href="#" className="hover:underline">{item}</a>
                </li>
              ))}
            </ul>
          </div>

          {/* Training Centres */}
          <div className="break-words">
            <h3 className="font-bold text-base mb-4">TRAINING CENTRES</h3>
            <ul className="space-y-2 text-sm">
              {[
                'CATC, Allahabad',
                'Indian Aviation Academy',
                'AAI Fire Training Institute, New Delhi',
                'AAI Fire Training Institute, Kolkata',
                'Aviation Security Training Institute (ASTI)',
              ].map(item => (
                <li key={item}>
                  <a href="#" className="hover:underline break-words">{item}</a>
                </li>
              ))}
            </ul>
          </div>

          {/* Corporate Address */}
          <div className="break-words">
            <h3 className="font-bold text-base mb-4">CORPORATE ADDRESS</h3>
            <p className="text-sm leading-relaxed">
              Airports Authority of India,<br />
              Rajiv Gandhi Bhawan,<br />
              Safdarjung Airport,<br />
              New Delhi – 110003<br />
              Ph: +91-11-24632950
            </p>
            <img
              src="/good-governance.png"
              alt="Good Governance"
              className="h-12 mt-4 max-w-full"
            />
          </div>
        </div>

        {/* Bottom Footer */}
        <div className="border-t border-blue-700 pt-6 text-xs text-center space-y-3 break-words">
          <div className="flex flex-wrap justify-center gap-x-4 gap-y-2">
            {[
              'Right to Information',
              'Contact Us',
              'Feedback',
              'AAI Email',
              'Sitemap',
              'Privacy Policy',
              'Disclaimer',
              'Terms and Conditions',
              'FAQs',
            ].map(item => (
              <a key={item} href="#" className="hover:underline">
                {item}
              </a>
            ))}
          </div>

          <div>© 2025 Airports Authority of India. All Rights Reserved.</div>
          <div>Designed by AAI | ISI Visits Count: 23621570</div>
          <div className="px-2">
            Best viewed on Chrome, Firefox, Safari, IE11+ | Resolution 1280×720 |
            Last Updated: 30 Jan 2025
          </div>
        </div>

      </div>
    </footer>
  )
}

export default Footer
