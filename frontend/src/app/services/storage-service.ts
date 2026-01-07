import { Injectable } from '@angular/core';
import { ResultItem } from '../models/result.type';
import { HistoryItem } from '../models/history.type';
import { Query } from '../models/query.type';
import { Result } from 'postcss';

@Injectable({
  providedIn: 'root',
})
export class StorageService {

  private STORAGE_KEY = 'history';

  private save(data: HistoryItem[]): void {
    localStorage.setItem(this.STORAGE_KEY, JSON.stringify(data));
  }

  private load(): HistoryItem[]{
    const store = localStorage.getItem(this.STORAGE_KEY);
    return store ? JSON.parse(store) : [];
  }

  getAll(): Record<string, any> {
    return this.load();
  }

  addQuery(query: Query): void {
    const store = this.load();
    store.push({query, results: []});
    this.save(store);
  }

  addResults(queryID: string, results: ResultItem[]): void {
    const store = this.load();
    const entry = store.find(e => e.query.id === queryID);

    if(!entry) return;

    entry.results.push(...results.flat());
    this.save(store);
  }

  getByQueryID(id: string): HistoryItem | undefined {
    return this.load().find(e => e.query.id === id);
  }
}
