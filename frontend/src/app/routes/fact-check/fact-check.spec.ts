import { ComponentFixture, TestBed } from '@angular/core/testing';

import { FactCheck } from './fact-check';

describe('FactCheck', () => {
  let component: FactCheck;
  let fixture: ComponentFixture<FactCheck>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [FactCheck]
    })
    .compileComponents();

    fixture = TestBed.createComponent(FactCheck);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
