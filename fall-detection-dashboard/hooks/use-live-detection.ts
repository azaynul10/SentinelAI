"use client"

import { useState, useRef, useCallback, useEffect } from "react"
import Webcam from "react-webcam"
import { useMutation } from "@tanstack/react-query"

export type DetectionResult = {
    fall_detected: boolean
    state: string
    confidence_score: number
    detection_method: string
    annotated_frame: string // base64
    paused: boolean
    fps?: number; // Optional FPS from backend
}

export function useLiveDetection() {
    const webcamRef = useRef<Webcam>(null)
    const videoRef = useRef<HTMLVideoElement>(null) // NEW: Support for local video file

    const [data, setData] = useState<DetectionResult | null>(null)
    const [isProcessing, setIsProcessing] = useState(false)
    const [sourceMode, setSourceMode] = useState<"camera" | "video">("camera") // Toggle state
    const intervalRef = useRef<NodeJS.Timeout | null>(null)

    // API Mutation
    const detectMutation = useMutation({
        mutationFn: async (frameSrc: string) => {
            const res = await fetch("http://localhost:5000/detect_fall", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ frame: frameSrc }),
            })
            if (!res.ok) throw new Error("API Error")
            return res.json() as Promise<DetectionResult>
        },
        onSuccess: (result) => {
            setData(result)
            setIsProcessing(false)
        },
        onError: () => {
            setIsProcessing(false)
        },
    })

    // Helper to capture frame from HTMLVideoElement
    const captureVideoFrame = (video: HTMLVideoElement): string | null => {
        if (video.paused || video.ended) return null;

        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        if (!ctx) return null;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL("image/jpeg", 0.8);
    }

    // Reset data when switching sources
    useEffect(() => {
        setData(null)
        setIsProcessing(false)
    }, [sourceMode])

    const captureAndSend = useCallback(() => {
        if (isProcessing) return

        let imageSrc: string | null | undefined = null;

        if (sourceMode === "camera" && webcamRef.current) {
            imageSrc = webcamRef.current.getScreenshot()
        } else if (sourceMode === "video" && videoRef.current) {
            imageSrc = captureVideoFrame(videoRef.current)
        }

        if (imageSrc) {
            setIsProcessing(true)
            detectMutation.mutate(imageSrc)
        }
    }, [detectMutation, isProcessing, sourceMode])

    // Polling Loop
    useEffect(() => {
        intervalRef.current = setInterval(captureAndSend, 100) // 10 FPS
        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current)
        }
    }, [captureAndSend])

    return {
        webcamRef,
        videoRef,
        data,
        isConnected: !detectMutation.isError,
        sourceMode,
        setSourceMode
    }
}
