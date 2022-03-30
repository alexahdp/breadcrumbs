import { List } from "../linked-list";

// it must be enough large to avoid collisions
const BUCKET_SIZE = 100_000;
const PRIME = 31;

export class HashTable<T> {
  private readonly buckets: List<{key: string, value: T}>[];

  constructor(size: number = BUCKET_SIZE) {
    this.buckets = Array(size).fill(0).map(() => new List());
  }

  hash(key: string): number {
    const hash = key.split('').reduce((o: number, v: string, i) => {
      return o + v.charCodeAt(0) * PRIME^(i);
    }, 0);
    return hash % this.buckets.length;
  }

  set(key: string, value: T) {
    const hash = this.hash(key);
    const list = this.buckets[hash];
    const node = list.search(node => node.value?.key === key);
    if (node) {
      node.value.value = value;
    } else {
      list.add({key, value});
    }
  }

  delete(key: string): boolean {
    const hash = this.hash(key);
    const list = this.buckets[hash];
    return list.remove(node => node.value.key === key);
  }

  get(key: string): unknown {
    const hash = this.hash(key);
    const list = this.buckets[hash];
    const node = list.search(node => node.value.key === key);
    return node ? node.value : null;
  }

  has(key: string) {
    const hash = this.hash(key);
    const list = this.buckets[hash];
    return !!list.search(node => node.value.key === key);
  }

  getKeys(): string[] {
    const result = [];
    for (const bucket of this.buckets) {
      for (const item of bucket.traverse()) {
        result.push(item.key);
      }
    }
    return result;
  }

  getValues(): T[] {
    const result = [];
    for (const bucket of this.buckets) {
      for (const item of bucket.traverse()) {
        result.push(item.value);
      }
    }
    return result;
  }
}
