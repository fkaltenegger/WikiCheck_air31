import { inject, Injectable, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { ResultItem } from '../models/result.type';

@Injectable({
  providedIn: 'root',
})
export class CheckService {
  data = signal<ResultItem[]>([]);
  loading = signal(false);
  error = signal<string | null>(null);

  constructor(private http: HttpClient) {}

  checkFactApi(method: string, query: string, ce: boolean, response_language: string) {
    this.loading.set(true);
    this.error.set(null);

    const body = { method, query, ce, response_language };

    this.http.post<ResultItem[]>('http://localhost:8000/check', body)
      .subscribe({
        next: (response: ResultItem[]) => {
          this.data.set(response);
          this.loading.set(false);
        },
        error: () => {
          this.error.set('Failed to load data');
          this.loading.set(false);
        }
      });
  }

  getEvaluation() {
    this.loading.set(true);
    this.error.set(null);

    const body = null;

    this.http.post<any>('http://localhost:8000/evaluation', body)
      .subscribe({
        next: (response: any) => {
          console.log(response);
          this.data.set(response);
          this.loading.set(false);
        },
        error: () => {
          this.error.set('Failed to load data');
          this.loading.set(false);
        }
      });
  }

}




// Title, Url, Parapgrah where its most matchable

// Top 3 Results

