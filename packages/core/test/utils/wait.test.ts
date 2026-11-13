import { expect, test, vi } from 'vitest';
import { waitEvent } from '#/utils/wait';

test('waitEvent filters generic events before resolving and unsubscribing', async () => {
  let emit: (event: string) => void = () => {};
  const unsubscribe = vi.fn();
  const match = vi.fn((event: string) => event === 'ready');
  const result = waitEvent<string>((handler) => {
    emit = handler;
    return unsubscribe;
  }, { match });

  emit('pending');
  expect(match).toHaveBeenCalledWith('pending');
  expect(unsubscribe).not.toHaveBeenCalled();
  emit('ready');
  await expect(result).resolves.toBe('ready');
  expect(match).toHaveBeenCalledTimes(2);
  expect(unsubscribe).toHaveBeenCalledOnce();
});
