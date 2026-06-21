import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  experimental: {
    // Tree-shake barrel imports from export-heavy client packages.
    optimizePackageImports: ["@xyflow/react", "@tanstack/react-query", "lucide-react"],
  },
};

export default nextConfig;
