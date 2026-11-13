import { describe, expect, it } from 'vitest';

import { createMessageAccumulator, fromCorePart, toCorePart } from '#/message';

describe('core message conversion', () => {
  it('preserves MIME parameters on media data URLs', () => {
    const url = 'data:video/mp4;codecs=avc1;base64,AQID';
    const core = toCorePart({ type: 'video_url', videoUrl: { url } });
    const legacy = fromCorePart(core);
    if (legacy.type !== 'video_url') throw new Error('expected a video part');
    expect(legacy.videoUrl.url).toBe(url);
  });
});

describe('message accumulation', () => {
  it('preserves both deferred reasoning keys when a tool call flushes text', () => {
    const accumulator = createMessageAccumulator();
    accumulator.push({ type: 'text', text: 'hello' });
    accumulator.push({ type: 'think', think: '', reasoningKey: 'reasoning_content' });
    accumulator.push({ type: 'think', think: '', reasoningKey: 'reasoning' });
    accumulator.push({ type: 'function', id: 'call_1', name: 'lookup', arguments: '{}' });
    expect(accumulator.finish().content).toEqual([
      { type: 'text', text: 'hello' },
      { type: 'think', think: '', reasoningKey: 'reasoning_content' },
      { type: 'think', think: '', reasoningKey: 'reasoning' },
    ]);
  });
});
