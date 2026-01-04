import { TestBed } from '@angular/core/testing';

import { Check } from './check';

describe('Check', () => {
  let service: Check;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(Check);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
