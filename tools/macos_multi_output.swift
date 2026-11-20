// macos_multi_output.swift - creates a "Dance Output" multi-output device (headphones/speakers +
// BlackHole) and makes it the default output, so you hear the music AND tools/dance_live.py
// captures it from BlackHole.
//
//     swift tools/macos_multi_output.swift                  # first output device whose name has "AirPods"
//     swift tools/macos_multi_output.swift "MacBook Pro Speakers"
//     swift tools/macos_multi_output.swift --restore "AirPods"   # back to plain AirPods, removes Dance Output
//
// The device is created by this process with kAudioAggregateDeviceIsPrivateKey = false, so it stays
// after the script exits (visible in Audio MIDI Setup). Re-running replaces it.
import CoreAudio
import Foundation

let NAME = "Dance Output"
let UID = "com.lerobot.dance.multioutput"

func prop(_ sel: AudioObjectPropertySelector, _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal)
    -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: sel, mScope: scope, mElement: kAudioObjectPropertyElementMain)
}

func devices() -> [AudioDeviceID] {
    var addr = prop(kAudioHardwarePropertyDevices)
    var size: UInt32 = 0
    AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size)
    var ids = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
    AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &ids)
    return ids
}

func string(_ id: AudioDeviceID, _ sel: AudioObjectPropertySelector) -> String {
    var addr = prop(sel)
    var value: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    guard AudioObjectGetPropertyData(id, &addr, 0, nil, &size, &value) == noErr, let v = value else { return "" }
    return v.takeRetainedValue() as String
}

func outputChannels(_ id: AudioDeviceID) -> Int {
    var addr = prop(kAudioDevicePropertyStreamConfiguration, kAudioObjectPropertyScopeOutput)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(id, &addr, 0, nil, &size) == noErr, size > 0 else { return 0 }
    let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: 16)
    defer { raw.deallocate() }
    guard AudioObjectGetPropertyData(id, &addr, 0, nil, &size, raw) == noErr else { return 0 }
    let list = UnsafeMutableAudioBufferListPointer(raw.assumingMemoryBound(to: AudioBufferList.self))
    return list.reduce(0) { $0 + Int($1.mNumberChannels) }
}

func findOutput(_ needle: String) -> AudioDeviceID? {
    devices().first { string($0, kAudioObjectPropertyName).localizedCaseInsensitiveContains(needle) && outputChannels($0) > 0 }
}

func setDefaultOutput(_ id: AudioDeviceID) {
    var dev = id
    for sel in [kAudioHardwarePropertyDefaultOutputDevice, kAudioHardwarePropertyDefaultSystemOutputDevice] {
        var addr = prop(sel)
        AudioObjectSetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil,
                                   UInt32(MemoryLayout<AudioDeviceID>.size), &dev)
    }
}

func destroyExisting() {
    for id in devices() where string(id, kAudioDevicePropertyDeviceUID) == UID {
        AudioHardwareDestroyAggregateDevice(id)
    }
}

var args = Array(CommandLine.arguments.dropFirst())
if args.first == "--restore" {
    let target = args.count > 1 ? args[1] : "AirPods"
    guard let main = findOutput(target) else { print("no output device matching \"\(target)\""); exit(1) }
    setDefaultOutput(main)
    destroyExisting()
    print("default output: \(string(main, kAudioObjectPropertyName)); \(NAME) removed")
    exit(0)
}
let target = args.first ?? "AirPods"
guard let main = findOutput(target) else {
    print("no output device matching \"\(target)\" (connected?). Outputs: "
          + devices().filter { outputChannels($0) > 0 }.map { string($0, kAudioObjectPropertyName) }.joined(separator: ", "))
    exit(1)
}
guard let hole = findOutput("BlackHole") else { print("BlackHole not found: brew install --cask blackhole-2ch"); exit(1) }
let mainUID = string(main, kAudioDevicePropertyDeviceUID)
let holeUID = string(hole, kAudioDevicePropertyDeviceUID)

destroyExisting()
let desc: [String: Any] = [
    kAudioAggregateDeviceNameKey: NAME,
    kAudioAggregateDeviceUIDKey: UID,
    kAudioAggregateDeviceIsStackedKey: 1,          // 1 = multi-output (same audio to every sub-device)
    kAudioAggregateDeviceIsPrivateKey: 0,          // keep it after this process exits
    kAudioAggregateDeviceMainSubDeviceKey: mainUID, // clock source: the headphones
    kAudioAggregateDeviceSubDeviceListKey: [
        [kAudioSubDeviceUIDKey: mainUID],
        [kAudioSubDeviceUIDKey: holeUID, kAudioSubDeviceDriftCompensationKey: 1],
    ],
]
var agg: AudioDeviceID = 0
let err = AudioHardwareCreateAggregateDevice(desc as CFDictionary, &agg)
guard err == noErr else { print("AudioHardwareCreateAggregateDevice failed: \(err)"); exit(1) }
Thread.sleep(forTimeInterval: 0.5)  // let coreaudiod publish it
setDefaultOutput(agg)
print("default output: \(NAME) = \(string(main, kAudioObjectPropertyName)) (you hear) + BlackHole 2ch (dance_live listens)")
