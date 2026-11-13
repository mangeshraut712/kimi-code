import { describe, expect, it } from 'vitest';
import { createMessageAccumulator, createMessageAccumulator as createLlmMessageAccumulator } from '#/llm/message';
import { createDecoder as createOpenAIDecoder } from '#/llm/builtin/protocol/openai/decode';
import { lowerMessage } from '#/llm/builtin/protocol/openai/lower';
import type { DecodeSink } from '#/llm/protocol/protocol';

function chatCompletionChunks(
  deltas: readonly Record<string, unknown>[] = [{ role: 'assistant', content: 'hi' }],
): Record<string, unknown>[] {
  return deltas.map((delta) => ({
    id: 'chatcmpl-1',
    object: 'chat.completion.chunk',
    created: 0,
    model: 'test-model',
    choices: [{ index: 0, delta, finish_reason: null }],
  }));
}

function feed(chunks: readonly Record<string, unknown>[], accumulator: ReturnType<typeof createMessageAccumulator>) {
  const decoder = createOpenAIDecoder({
    onDelta: part => { accumulator.push(part); },
    onFinish: () => {}, onMessageId: () => {}, onUsage: () => {}, onError: () => {},
    onToolCallEnd: index => { accumulator.toolCallEnd(index); },
  });
  for (const chunk of chunks) decoder.push(chunk);
  decoder.end();
}

describe('empty reasoning stream accumulation', () => {
  it.each(['reasoning_content', 'reasoning'])('drops repeated empty %s deltas without splitting text', key => {
    const accumulator = createMessageAccumulator();
    feed(chatCompletionChunks([{ [key]: 'plan' }, { content: 'a', [key]: '' }, { content: 'b', [key]: '' }, { content: 'c', [key]: '' }]), accumulator);
    expect(accumulator.finish().content).toEqual([
      { type: 'think', think: 'plan', reasoningKey: key },
      { type: 'text', text: 'abc' },
    ]);
  });

  it.each(['reasoning_content', 'reasoning'])('detects the %s dialect from empty deltas', key => {
    const detected: string[] = [];
    const accumulator = createMessageAccumulator();
    const decoder = createOpenAIDecoder({
      onDelta: part => { accumulator.push(part); },
      onFinish: () => {}, onMessageId: () => {}, onUsage: () => {}, onError: () => {},
      onToolCallEnd: index => { accumulator.toolCallEnd(index); },
    }, undefined, value => { detected.push(value); });
    for (const chunk of chatCompletionChunks([
      { content: 'hello' },
      { [key]: '', content: ' world' },
    ])) decoder.push(chunk);
    decoder.end();
    expect(detected).toEqual([key]);
    expect(accumulator.finish().content).toEqual([{ type: 'text', text: 'hello world' }]);
  });

  it('keeps whitespace-only reasoning deltas between non-empty chunks', () => {
    const chunks = chatCompletionChunks([{ reasoning: 'First' }, { reasoning: '\n\n' }, { reasoning: 'Second' }]);
    const accumulator = createMessageAccumulator();
    feed(chunks, accumulator);
    expect(accumulator.finish().content).toEqual([
      { type: 'think', think: 'First\n\nSecond', reasoningKey: 'reasoning' },
    ]);
  });

  it('keeps an empty marker deferred by a standalone delta when a tool call repeats it', () => {
    const chunks = chatCompletionChunks([
        { content: 'hello' },
        { reasoning: '' },
        { content: ' world' },
        {
          reasoning: '',
          tool_calls: [
            { index: 0, id: 'call_1', type: 'function', function: { name: 'lookup', arguments: '{}' } },
          ],
        },
      ]);
    const accumulator = createMessageAccumulator();
    feed(chunks, accumulator);
    const message = accumulator.finish();
    expect(message.content).toEqual([
      { type: 'text', text: 'hello world' },
      { type: 'think', think: '', reasoningKey: 'reasoning' },
    ]);
  });

  it('deduplicates both standalone empty keys after a tool header flushes them', () => {
    const accumulator = createMessageAccumulator();
    feed(chatCompletionChunks([
      { content: 'hello' },
      { reasoning_content: '', reasoning: '' },
      { tool_calls: [{ index: 0, id: 'call_1', function: { name: 'lookup', arguments: '{}' } }] },
      { reasoning_content: '', reasoning: '' },
    ]), accumulator);
    expect(accumulator.finish().content).toEqual([
      { type: 'text', text: 'hello' },
      { type: 'think', think: '', reasoningKey: 'reasoning_content' },
      { type: 'think', think: '', reasoningKey: 'reasoning' },
    ]);
  });

  it('deduplicates empty markers after a tool header flushes one', () => {
    const chunks = chatCompletionChunks([
        { content: 'hello' },
        {
          reasoning: '',
          tool_calls: [
            {
              index: 0,
              id: 'call_1',
              type: 'function',
              function: { name: 'lookup', arguments: '' },
            },
          ],
        },
        {
          reasoning: '',
          tool_calls: [{ index: 0, function: { arguments: '{"q":"x"}' } }],
        },
      ]);
    const accumulator = createMessageAccumulator();
    feed(chunks, accumulator);
    const message = accumulator.finish();
    expect(message.content).toEqual([
      { type: 'text', text: 'hello' },
      { type: 'think', think: '', reasoningKey: 'reasoning' },
    ]);
    expect(message.toolCalls).toEqual([
      expect.objectContaining({ id: 'call_1', name: 'lookup', arguments: '{"q":"x"}' }),
    ]);
  });

  it('keeps a replayed empty marker eligible when the same chunk only routes arguments', () => {
    const chunks = chatCompletionChunks([
        {
          tool_calls: [
            {
              index: 0,
              id: 'call_1',
              type: 'function',
              function: { name: 'lookup', arguments: '' },
            },
          ],
        },
        { content: 'hello' },
        {
          reasoning: '',
          content: ' world',
          tool_calls: [{ index: 0, function: { arguments: '{"q":"x"}' } }],
        },
        { content: '!' },
        {
          reasoning: '',
          tool_calls: [
            {
              index: 1,
              id: 'call_2',
              type: 'function',
              function: { name: 'lookup', arguments: '{}' },
            },
          ],
        },
      ]);
    const accumulator = createMessageAccumulator();
    feed(chunks, accumulator);
    const message = accumulator.finish();
    expect(message.content).toEqual([
      { type: 'text', text: 'hello world!' },
      { type: 'think', think: '', reasoningKey: 'reasoning' },
    ]);
    expect(message.toolCalls).toEqual([
      expect.objectContaining({ id: 'call_1', name: 'lookup', arguments: '{"q":"x"}' }),
      expect.objectContaining({ id: 'call_2', name: 'lookup', arguments: '{}' }),
    ]);
    const [wire] = lowerMessage(message, {
      reasoningKey: 'reasoning_content',
      preserveThinking: false,
      toolMessageConversion: undefined,
    });
    expect(wire).toMatchObject({
      role: 'assistant',
      content: 'hello world!',
      reasoning: '',
    });
  });

  it('keeps an empty reasoning marker when argument deltas route to a flushed tool', () => {
    const chunks = chatCompletionChunks([
        {
          tool_calls: [
            {
              index: 0,
              id: 'call_1',
              type: 'function',
              function: { name: 'lookup', arguments: '' },
            },
          ],
        },
        {
          content: 'hello',
          tool_calls: [{ index: 0, function: { arguments: '{"q":"x"}' } }],
        },
        { reasoning: '', content: ' world' },
        {
          reasoning: '',
          tool_calls: [
            {
              index: 1,
              id: 'call_2',
              type: 'function',
              function: { name: 'lookup', arguments: '{}' },
            },
          ],
        },
      ]);
    const accumulator = createMessageAccumulator();
    feed(chunks, accumulator);
    const message = accumulator.finish();
    expect(message.content).toEqual([
      { type: 'text', text: 'hello world' },
      { type: 'think', think: '', reasoningKey: 'reasoning' },
    ]);
    expect(message.toolCalls).toEqual([
      expect.objectContaining({ id: 'call_1', name: 'lookup', arguments: '{"q":"x"}' }),
      expect.objectContaining({ id: 'call_2', name: 'lookup', arguments: '{}' }),
    ]);
    const [wire] = lowerMessage(message, {
      reasoningKey: 'reasoning_content',
      preserveThinking: false,
      toolMessageConversion: undefined,
    });
    expect(wire).toMatchObject({
      role: 'assistant',
      content: 'hello world',
      reasoning: '',
    });
    expect(wire?.tool_calls).toEqual([
      expect.objectContaining({
        id: 'call_1',
        function: { name: 'lookup', arguments: '{"q":"x"}' },
      }),
      expect.objectContaining({
        id: 'call_2',
        function: { name: 'lookup', arguments: '{}' },
      }),
    ]);
  });

  it('keeps one empty reasoning marker when text clears the first and a tool call repeats it', () => {
    const chunks = chatCompletionChunks([
        { content: 'hello' },
        { content: ' world', reasoning: '' },
        {
          reasoning: '',
          tool_calls: [
            {
              index: 0,
              id: 'call_1',
              type: 'function',
              function: { name: 'lookup', arguments: '{}' },
            },
          ],
        },
      ]);
    const accumulator = createMessageAccumulator();
    feed(chunks, accumulator);
    const message = accumulator.finish();
    expect(message.content).toEqual([
      { type: 'text', text: 'hello world' },
      { type: 'think', think: '', reasoningKey: 'reasoning' },
    ]);
    expect(message.toolCalls).toEqual([
      expect.objectContaining({ name: 'lookup', arguments: '{}' }),
    ]);
    const [wire] = lowerMessage(message, {
      reasoningKey: 'reasoning_content',
      preserveThinking: false,
      toolMessageConversion: undefined,
    });
    expect(wire).toMatchObject({
      role: 'assistant',
      content: 'hello world',
      reasoning: '',
    });
    expect(wire?.tool_calls).toEqual([
      expect.objectContaining({
        function: { name: 'lookup', arguments: '{}' },
      }),
    ]);
  });

  it('keeps one empty reasoning marker per key when a delta with text and a tool header repeats both', () => {
    const chunks = chatCompletionChunks([
        { content: 'hello' },
        {
          content: ' world',
          reasoning_content: '',
          reasoning: '',
          tool_calls: [
            {
              index: 0,
              id: 'call_1',
              type: 'function',
              function: { name: 'lookup', arguments: '{}' },
            },
          ],
        },
      ]);
    const accumulator = createMessageAccumulator();
    feed(chunks, accumulator);
    const message = accumulator.finish();
    expect(message.content).toEqual([
      { type: 'text', text: 'hello world' },
      { type: 'think', think: '', reasoningKey: 'reasoning_content' },
      { type: 'think', think: '', reasoningKey: 'reasoning' },
    ]);
    const [wire] = lowerMessage(message, {
      reasoningKey: 'reasoning_content',
      preserveThinking: false,
      toolMessageConversion: undefined,
    });
    expect(wire).toMatchObject({
      role: 'assistant',
      content: 'hello world',
      reasoning_content: '',
      reasoning: '',
    });
  });
});

describe('openai decoder tool-call ordering', () => {
  function recordingSink(): { sink: DecodeSink; events: string[] } {
    const events: string[] = [];
    const sink: DecodeSink = {
      onDelta: (part) => {
        if (part.type === 'function') events.push(`function:${part.name}:${part.arguments ?? ''}`);
        else if (part.type === 'tool_call_part') events.push(`args:${String(part.index)}:${part.argumentsPart ?? ''}`);
        else if (part.type === 'think') events.push(`think:${part.reasoningKey ?? ''}:${part.think}`);
        else if (part.type === 'text') events.push(`text:${part.text}`);
      },
      onFinish: () => {},
      onMessageId: () => {},
      onUsage: () => {},
      onError: () => {},
      onToolCallEnd: (index) => events.push(`end:${String(index)}`),
    };
    return { sink, events };
  }

  it('emits each tool call before ending it when one delta carries two headers', () => {
    const { sink, events } = recordingSink();
    const decoder = createOpenAIDecoder(sink);
    decoder.push({
      choices: [
        {
          delta: {
            tool_calls: [
              { index: 0, id: 'a', function: { name: 'first', arguments: '{}' } },
              { index: 1, id: 'b', function: { name: 'second', arguments: '{}' } },
            ],
          },
          finish_reason: 'tool_calls',
        },
      ],
    });
    expect(events).toEqual(['function:first:{}', 'end:0', 'function:second:{}', 'end:1']);
  });

  it('emits the active call\'s final arguments before a new header ends it', () => {
    const { sink, events } = recordingSink();
    const decoder = createOpenAIDecoder(sink);
    decoder.push({
      choices: [
        {
          delta: { tool_calls: [{ index: 0, id: 'a', function: { name: 'first', arguments: '{"q":' } }] },
          finish_reason: null,
        },
      ],
    });
    decoder.push({
      choices: [
        {
          delta: {
            tool_calls: [
              { index: 0, function: { arguments: '1}' } },
              { index: 1, id: 'b', function: { name: 'second', arguments: '{}' } },
            ],
          },
          finish_reason: 'tool_calls',
        },
      ],
    });
    expect(events).toEqual([
      'function:first:{"q":',
      'args:0:1}',
      'end:0',
      'function:second:{}',
      'end:1',
    ]);
  });

  it('keeps both deferred empty reasoning keys in the llm accumulator', () => {
    const accumulator = createLlmMessageAccumulator();
    const decoder = createOpenAIDecoder({
      onDelta: (part) => {
        accumulator.push(part);
      },
      onFinish: () => {},
      onMessageId: () => {},
      onUsage: () => {},
      onError: () => {},
      onToolCallEnd: (index) => accumulator.toolCallEnd(index),
    });
    decoder.push({ choices: [{ delta: { content: 'hello' }, finish_reason: null }] });
    decoder.push({
      choices: [
        {
          delta: {
            content: ' world',
            reasoning_content: '',
            ...{ reasoning: '' },
            tool_calls: [{ index: 0, id: 'a', function: { name: 'lookup', arguments: '{}' } }],
          },
          finish_reason: 'tool_calls',
        },
      ],
    });
    const message = accumulator.finish();
    expect(message.content).toEqual([
      { type: 'text', text: 'hello world' },
      { type: 'think', think: '', reasoningKey: 'reasoning_content' },
      { type: 'think', think: '', reasoningKey: 'reasoning' },
    ]);
    expect(message.toolCalls).toEqual([expect.objectContaining({ name: 'lookup', arguments: '{}' })]);
  });
});
