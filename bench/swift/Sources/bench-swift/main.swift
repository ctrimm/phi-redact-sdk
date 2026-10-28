// Swift bench harness. Prints one JSON line per invocation.
//
//   bench-swift scan|rules|features
//   bench-swift batch|batch-features
//
// Driven by bench/run_bench.py, which interleaves languages
// round-by-round; this harness executes BENCH_REPS measured passes
// (after BENCH_WARMUPS discarded passes) in-process and reports raw
// per-note samples plus SHA-256 digests of its inputs and outputs in
// the payload format shared with the Python, TS and Rust harnesses
// (see bench/bench_python.py):
//
//   per redact result: redacted text, then one line per entity
//   "start,end,label,score,source,placeholder", then one line per
//   review candidate "Rstart,end,label,score,source"; score is ""
//   when the span carries none, else formatted %.4f. Payloads are
//   joined per note and hashed incrementally with a NUL separator.
//
// Corpus paths resolve against BENCH_SDK_ROOT (set by the driver),
// falling back to the current working directory.
//
// The kit's neural path is Core ML, which does not exist on Linux —
// there is deliberately no "neural" mode here.

import Foundation
import PhiRedactKit

// MARK: - SHA-256
//
// The digest format is shared across harnesses; PhiRedactKit's own
// SHA256 type is internal, so the bench carries its own. Verified
// against the FIPS 180-4 example vectors in a pre-flight check
// (mode "selftest").

struct SHA256 {
    private static let k: [UInt32] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ]
    private var h: [UInt32] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ]
    private var buffer: [UInt8] = []
    private var totalLength: UInt64 = 0

    mutating func update(_ bytes: [UInt8]) {
        totalLength &+= UInt64(bytes.count)
        buffer.append(contentsOf: bytes)
        while buffer.count >= 64 {
            process(Array(buffer.prefix(64)))
            buffer.removeFirst(64)
        }
    }

    private mutating func process(_ block: [UInt8]) {
        var w = [UInt32](repeating: 0, count: 64)
        for i in 0..<16 {
            w[i] = UInt32(block[4 * i]) << 24 | UInt32(block[4 * i + 1]) << 16
                | UInt32(block[4 * i + 2]) << 8 | UInt32(block[4 * i + 3])
        }
        for i in 16..<64 {
            let s0 = w[i - 15].rotatedRight(7) ^ w[i - 15].rotatedRight(18) ^ (w[i - 15] >> 3)
            let s1 = w[i - 2].rotatedRight(17) ^ w[i - 2].rotatedRight(19) ^ (w[i - 2] >> 10)
            w[i] = w[i - 16] &+ s0 &+ w[i - 7] &+ s1
        }
        var (a, b, c, d, e, f, g, hh) = (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7])
        for i in 0..<64 {
            let s1 = e.rotatedRight(6) ^ e.rotatedRight(11) ^ e.rotatedRight(25)
            let ch = (e & f) ^ (~e & g)
            let t1 = hh &+ s1 &+ ch &+ Self.k[i] &+ w[i]
            let s0 = a.rotatedRight(2) ^ a.rotatedRight(13) ^ a.rotatedRight(22)
            let maj = (a & b) ^ (a & c) ^ (b & c)
            let t2 = s0 &+ maj
            hh = g; g = f; f = e; e = d &+ t1; d = c; c = b; b = a; a = t1 &+ t2
        }
        h[0] = h[0] &+ a; h[1] = h[1] &+ b; h[2] = h[2] &+ c; h[3] = h[3] &+ d
        h[4] = h[4] &+ e; h[5] = h[5] &+ f; h[6] = h[6] &+ g; h[7] = h[7] &+ hh
    }

    func hexDigest() -> String {
        var copy = self
        let bitLength = totalLength &* 8
        copy.update([0x80])
        while copy.buffer.count != 56 { copy.update([0]) }
        var lenBytes = [UInt8](repeating: 0, count: 8)
        for i in 0..<8 { lenBytes[7 - i] = UInt8((bitLength >> (8 * i)) & 0xff) }
        // Append length without touching totalLength accounting.
        copy.buffer.append(contentsOf: lenBytes)
        while copy.buffer.count >= 64 {
            copy.process(Array(copy.buffer.prefix(64)))
            copy.buffer.removeFirst(64)
        }
        return copy.h.map { String(format: "%08x", $0) }.joined()
    }
}

extension UInt32 {
    func rotatedRight(_ n: UInt32) -> UInt32 { (self >> n) | (self << (32 - n)) }
}

// MARK: - Canonical payloads (must match bench_python.py byte-for-byte)

let posix = Locale(identifier: "en_US_POSIX")

func scoreStr(_ score: Double?, _ withScore: Bool) -> String {
    guard withScore, let s = score else { return "" }
    return String(format: "%.4f", locale: posix, s)
}

func spansPayload(_ spans: [Entity], _ withScore: Bool) -> String {
    spans.map {
        "\($0.start),\($0.end),\($0.label),\(scoreStr($0.score, withScore)),"
            + "\($0.source.rawValue),\($0.placeholder ?? "")"
    }.joined(separator: "\n")
}

func resultPayload(_ r: RedactionResult, _ withScore: Bool) -> String {
    var lines = [r.redacted]
    for e in r.entities {
        lines.append(
            "\(e.start),\(e.end),\(e.label),\(scoreStr(e.score, withScore)),"
                + "\(e.source.rawValue),\(e.placeholder ?? "")")
    }
    for c in r.review {
        lines.append(
            "R\(c.start),\(c.end),\(c.label),\(scoreStr(c.score, withScore)),"
                + "\(c.source.rawValue)")
    }
    return lines.joined(separator: "\n")
}

func digestPayloads(_ payloads: [String]) -> String {
    var h = SHA256()
    for p in payloads {
        h.update([UInt8](p.utf8))
        h.update([0])
    }
    return h.hexDigest()
}

func inputDigest(_ texts: [String]) -> String {
    var h = SHA256()
    for t in texts {
        h.update([UInt8](t.utf8))
        h.update([0x0A])
    }
    return h.hexDigest()
}

// MARK: - Corpus loading

func loadNotes(_ path: String) -> [String] {
    let content = try! String(contentsOfFile: path, encoding: .utf8)
    return content.split(separator: "\n", omittingEmptySubsequences: false)
        .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        .map { line in
            let obj = try! JSONSerialization.jsonObject(with: Data(line.utf8))
                as! [String: Any]
            return obj["note"] as! String
        }
}

// MARK: - Timing

func nowNs() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

func round4(_ x: Double) -> Double { (x * 10000).rounded() / 10000 }

func perNote(_ fn: (String) -> Void, notes: [String],
             warmups: Int, reps: Int) -> [Double] {
    for _ in 0..<warmups { for n in notes { fn(n) } }
    var samples: [Double] = []
    for _ in 0..<reps {
        for n in notes {
            let t0 = nowNs()
            fn(n)
            samples.append(round4(Double(nowNs() - t0) / 1e6))
        }
    }
    return samples
}

func emit(_ obj: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: obj)
    print(String(data: data, encoding: .utf8)!)
}

// MARK: - Main

let env = ProcessInfo.processInfo.environment
let args = CommandLine.arguments
let mode = args.count > 1 ? args[1] : ""

if mode == "selftest" {
    var h = SHA256()
    h.update([UInt8]("abc".utf8))
    let abc = h.hexDigest()
    let empty = SHA256().hexDigest()
    print("abc:   \(abc)")
    print("empty: \(empty)")
    print(abc == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        && empty == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ? "SHA-256 OK" : "SHA-256 BROKEN")
    exit(abc.hasPrefix("ba7816bf") && empty.hasPrefix("e3b0c442") ? 0 : 1)
}

let root = env["BENCH_SDK_ROOT"] ?? FileManager.default.currentDirectoryPath
let notes = loadNotes(root + "/e2e/parity/notes.jsonl")
let isBatch = mode.hasPrefix("batch")
let warmups = Int(env["BENCH_WARMUPS"] ?? "") ?? (isBatch ? 1 : 2)
let reps = Int(env["BENCH_REPS"] ?? "") ?? (isBatch ? 5 : 7)

let featureOptions = RedactionOptions(
    useNeural: false,
    anonymizeOperator: .hash,
    salt: "bench",
    denylist: ["Maria"],
    recognizers: [
        PatternRecognizer(label: "EMPLOYEE_ID", patterns: [#"EMP\d{6}"#], score: 0.9),
        PatternRecognizer(label: "REF_CODE", patterns: [#"[A-Z]{2}-\d{4}"#], score: 0.8),
    ],
    thresholds: ["REF_CODE": 0.9],
    reviewMargin: 0.2
)
let plainOptions = RedactionOptions(useNeural: false)
let redactor = Redactor()

switch mode {
case "scan":
    // Digest pass first (untimed full pass -> steady state).
    let digest = digestPayloads(notes.map { spansPayload(Rules.scan($0), true) })
    let digestNs = digestPayloads(notes.map { spansPayload(Rules.scan($0), false) })
    let samples = perNote({ _ = Rules.scan($0) }, notes: notes,
                          warmups: warmups, reps: reps)
    emit(["language": "swift", "mode": mode,
          "input_digest": inputDigest(notes), "notes": notes.count,
          "samples_ms": samples, "digest": digest, "digest_noscore": digestNs])

case "rules", "features":
    let options = mode == "rules" ? plainOptions : featureOptions
    let fn = { (n: String) in _ = try! redactor.redact(n, options: options) }
    let digest = digestPayloads(notes.map {
        resultPayload(try! redactor.redact($0, options: options), true)
    })
    let digestNs = digestPayloads(notes.map {
        resultPayload(try! redactor.redact($0, options: options), false)
    })
    let samples = perNote(fn, notes: notes, warmups: warmups, reps: reps)
    emit(["language": "swift", "mode": mode,
          "input_digest": inputDigest(notes), "notes": notes.count,
          "samples_ms": samples, "digest": digest, "digest_noscore": digestNs])

case "batch", "batch-features":
    let options = mode == "batch" ? plainOptions : featureOptions
    let corpus = loadNotes(root + "/bench/corpus.jsonl")
    // Digest from a dedicated untimed pass (not tied to warmups).
    let digestResults = corpus.map { try! redactor.redact($0, options: options) }
    let digest = digestPayloads(digestResults.map { resultPayload($0, true) })
    let digestNs = digestPayloads(digestResults.map { resultPayload($0, false) })
    for _ in 0..<warmups { for t in corpus { _ = try! redactor.redact(t, options: options) } }
    var totals: [Double] = []
    for _ in 0..<reps {
        let t0 = nowNs()
        for t in corpus { _ = try! redactor.redact(t, options: options) }
        totals.append(round4(Double(nowNs() - t0) / 1e9))
    }
    emit(["language": "swift", "mode": mode, "docs": corpus.count,
          "totals_s": totals, "digest": digest, "digest_noscore": digestNs,
          "input_digest": inputDigest(corpus)])

default:
    FileHandle.standardError.write(Data("unknown mode \(mode)\n".utf8))
    exit(2)
}
