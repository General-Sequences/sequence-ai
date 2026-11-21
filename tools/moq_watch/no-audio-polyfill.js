// sequence-ai's console carries no audio polyfill: a robot's broadcasts have no audio, and libav.js's Opus build is LGPL.
// @moq/hang loads it only where a browser lacks AudioDecoder; that import then fails, and only audio would need it.
throw new Error("sequence-ai's teleop console carries no audio polyfill");
