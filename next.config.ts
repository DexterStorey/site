import type { NextConfig } from 'next'

const config: NextConfig = {
	redirects: async () => [
		{
			source: '/:path*',
			has: [{ type: 'host', value: 'www.dexterstorey.com' }],
			destination: 'https://dexterstorey.com/:path*',
			permanent: true
		},
		// Pages from the old site.
		{ source: '/:page(home|drafts|portfolio|catalog|storage1)', destination: '/', permanent: false }
	]
}

export default config
