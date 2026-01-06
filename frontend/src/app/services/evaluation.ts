import { HttpClient } from '@angular/common/http';
import { Injectable, signal } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class EvaluationService {

  data = signal<any>([]);
  loading = signal(false);
  error = signal<string | null>(null);

    constructor(private http: HttpClient) {}

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
